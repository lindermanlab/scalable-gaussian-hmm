"""Script for fitting HMM to killifish data via stochastic EM

There exists the option to profile the train_and_checkpoint function. A note on
how the code is written:
    In order to record memory usage when calling `memory_usage` on a
    python function, MUST specify stream=False (otherwise, returns None)
    and then manually write results to file
    NB: `memory_usage` only automatically writes to file if called by an
        external process, i.e. with the mprof command in the command line.
        When `mprof` called via a job, psutil has trouble finding the
        correct process id and throws a NoProcessFound error. It still seems
        to be able record it when submitted as non-interactive job, but
        program fails when submitted interactively
"""

import os
import argparse
from datetime import datetime
from memory_profiler import memory_usage
import time

import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from ssm_jax.hmm.models import GaussianHMM
from kf import (FishPCDataset, FishPCLoader,
                initialize_gaussian_hmm, CheckpointDataclass
               )

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']
fish_id = 'fish0_137'

# -------------------------------------
# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -------------------------------------

parser = argparse.ArgumentParser(description='Single-subject stochastic EM fit')
parser.add_argument(
    '--log_dir', type=str, default=TEMPDIR,
    help='Directory to log profiles.')
parser.add_argument(
    '--session_name', type=str, default=None,
    help='Identifying token,. Used for log and checkpoint files')
parser.add_argument(
    '--checkpoint_interval', type=int, default=-1,
    help='Number of epochs between intermediate checkpoints. If -1 (default), no checkpointing.')
parser.add_argument(
    '--mprof', action='store_true',
    help='If specified, profile memory usage with memory_profiler.')

parser.add_argument(
    '--hmm_init_method', type=str, default='random',
    choices=['random', 'kmeans'],
    help='HMM initialization method in the first epoch.')
parser.add_argument(
    '--seed', type=int, default=45212276,
    help='Initial RNG seed, for splitting data and intializing HMM.')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Number of batches loaded per iteration.')
parser.add_argument(
    '--seq_length', type=int, default=72000,
    help='Number of consecutive frames per sequence.')
parser.add_argument(
    '--train', type=float, default=0.8,
    help='If >=1, number of sequences of seq_length in dataset to train over. If [0, 1), fraction of sequences in dataset to train over.')
parser.add_argument(
    '--test', type=float, default=0.2,
    help='If >=1, number of sequences of seq_length in dataset to train over. If [0, 1), fraction of sequences in dataset to train over.')
parser.add_argument(
    '--epochs', type=int, default=10,
    help='Number of stochastic EM iterations to run')
parser.add_argument(
    '--states', type=int, default=20,
    help='Number of HMM states to fit')

parser.add_argument(
    '--debug_max_files', type=int, default=-1,
    help='FOR DEBUGGING: Maximum number of files (~days of recording) in directory to expose. Default: -1, expose all.')

def write_mprof(path: str, mem_usage: list, mode: str='w+') -> None:
    """Writes time-based memory profiler results to file."""
    with open(path, mode) as f:
        for res in mem_usage:
            f.writelines('MEM {} {}\n'.format(res[0], res[1]))
    return

# -------------------------------------

def setup_data(seed, batch_size, seq_length, split_sizes,
               starting_epoch=0, DEBUG_MAX_FILES=-1):
    """Construct training and validation dataloaders
    TODO Remove DEBUG_MAX_FILES argument (args.argparse, filepaths)
    """

    print("\n======================")
    print("Setting up dataset...")

    # We want dataset to be split the same way across epochs, to properly reserve
    # test data -> so seed_split as independent of starting_epoch. However, the
    # order of the minibatches should be shuffled every epoch. We can't control
    # this precisely with torch.Generator, however, we can at least make sure
    # it is different -> seed_dl_train folds in epoch information.
    seed_split, seed_dl, seed_debug = jr.split(seed, 3)
    seed_dl_train = jr.fold_in(seed_dl, starting_epoch)

    # TODO This is hard fixed to single subject for now
    fish_dir = os.path.join(DATADIR, fish_id)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])

    # TODO Hard limiting number of filepaths. Remove when done debugging
    if DEBUG_MAX_FILES > 0:
        print(f"!!! WARNING !!! Limiting total number of files loaded to {DEBUG_MAX_FILES}.")
        idxs = jr.permutation(seed_debug, len(filepaths))[:DEBUG_MAX_FILES]
        filepaths = [filepaths[i] for i in idxs]
    dataset = FishPCDataset(filepaths, return_labels=False)

    # Split dataset into a training set and a test.
    # test_dl does not need to be shuffled since we use full E-step to evaluate
    train_slices, test_slices = \
        dataset.split_seq(split_sizes, seed_split, seq_length,
                          step_size=1, drop_incomplete_seqs=True)
    train_dl = FishPCLoader(dataset, train_slices, batch_size,
                            drop_last=True, shuffle=True, seed=int(seed_dl_train[-1]))
    test_dl  = FishPCLoader(dataset, test_slices, batch_size,
                            drop_last=True, shuffle=False)
    
    print("Initialized training dataset with "
          + f"{len(train_slices):3d} sets of {seq_length/72000:.1f} hr sequences; "
          + f"{len(train_dl):3d} batches of {batch_size} sequences per batch")
    print("Initialized testing  dataset with "
          + f"{len(test_slices):3d} sets of {seq_length/72000:.1f} hr sequences; "
          + f"{len(test_dl):3d} batches of {batch_size} sequences per batch")
    print()

    return dataset, train_dl, test_dl

# -------------------------------------

def compute_exact_lp(hmm, dataloader):
    """Compute exact average log likelihood for a given HMM and dataset"""

    lp = 0.
    for batch_emissions in dataloader:
        batch_stats = hmm.e_step(batch_emissions)
        lp += batch_stats.marginal_loglik.sum()
    return lp

def train_and_checkpoint(train_dataloader,
                         hmm: GaussianHMM,
                         num_epochs: int,
                         checkpoint: CheckpointDataclass,
                         starting_epoch: int=0,
                         prev_train_lps=None,
                         prev_test_lps=None,
                         test_dataloader=None,
                         verbose=False,
                         ):
    """Fit HMM via stochastic EM, with intermediate checkpointing. After final
    epoch, HMM is automatically checkpointed a final time.

    Args
        train_dataloader (torch.utils.data.DataLoader): Iterable over training data
        hmm (GaussianHMM): GaussianHMM to train. May be partially trained.
        num_epochs (int): (Total) number of epochs to train.
        checkpoint (CheckpointDataclass): Container with checkpoint parameters.
        starting_epoch (int): Starting epoch of this training (i.e. hmm has already
            been partially trained; warm-start). Default: 0 (cold-start).
        prev_lps (array-like, length starting_epoch): Mean expected log
            probabilities from previous epochs, if HMM is being warm-started.
            Default: [] (cold-start, no previous training).
        test_dataloader (torch.utils.data.DataLoader): Iterable over test data.
            If not None, evaluate exact log probability on test data after each
            epoch. If None, do not calculate. Default: None
        verbose (boolean): If True, print log-probabilities for each minibatch
            in each epoch after each checkpoint interval. Default: False
    
    Returns
        all_lps (tuple of arrays): Mean expected log probabilities of model on
            train (and test) datasets. Array shapes: (num_epochs, num_batches)
        ckp_path (str): Path to last checkpoint file
    
    TODO Expose learning_rate
    """
    assert starting_epoch >= 0
    assert checkpoint.interval < num_epochs  
    assert starting_epoch < num_epochs

    # If no interval specified, run (remaining) number of epochs
    if checkpoint.interval < 1:
        checkpoint.interval = num_epochs - starting_epoch

    def _train_and_val():
        """Train model, and validate on test data after every epoch."""
        train_lps = onp.empty((checkpoint.interval, len(train_dataloader)))
        test_lps = onp.empty((checkpoint.interval, 1))
        for _epoch in range(checkpoint.interval):
            train_lp = hmm.fit_stochastic_em(
                train_dataloader, train_dataloader.total_emissions, num_epochs=1
            )

            train_lps[_epoch] = train_lp
            test_lps[_epoch] = compute_exact_lp(hmm, test_dataloader)

        return (train_lps, test_lps)

    if prev_train_lps is None:
        prev_train_lps = onp.empty((0,len(train_dataloader)))
    if prev_test_lps is None and test_dataloader:
        prev_test_lps = onp.empty((0,1))
    
    all_train_lps = prev_train_lps
    all_test_lps = prev_test_lps
    for last_epoch in range(starting_epoch, num_epochs, checkpoint.interval):
        if test_dataloader:
            train_lps, test_lps = _train_and_val()
            all_test_lps = onp.vstack([all_test_lps, test_lps])
        else:
            train_lps = hmm.fit_stochastic_em(train_dataloader,
                                              train_dataloader.total_emissions,
                                              num_epochs=checkpoint.interval)

        all_train_lps = onp.vstack([all_train_lps, train_lps])


        if verbose:
            for _epoch in range(len(train_lps)):
                print(f"\nEpoch {last_epoch+_epoch:2d}\n---------")
                print(f"Expected train:\n", train_lps[_epoch] / train_dataloader.total_emissions)
                if test_lps is not None:
                    print(f"\nExact test:\n", test_lps[_epoch / test_dataloader.total_emissions])

        this_epoch = last_epoch + checkpoint.interval
        
        ckp_path = checkpoint.save(hmm, this_epoch,
                                   all_train_lps=all_train_lps,
                                   all_test_lps=all_test_lps)

    return (all_train_lps, all_test_lps), ckp_path

def main():
    args = parser.parse_args()

    # Set session name for identification and set folder to store all outputs
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    session_name = timestamp if args.session_name is None else args.session_name
    log_dir = os.path.join(args.log_dir, session_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Output files will be logged to: {log_dir}\n")

    checkpointer = CheckpointDataclass(
        directory=os.path.join(log_dir, 'checkpoints'),
        interval=args.checkpoint_interval,
        keep=-1,
    )

    # ==========================================================================
    # Warm-start from last checkpoint, if available. Else, initialize hmm.
    hmm, prev_epoch, prev_train_lps, prev_test_lps, warm_ckp_path \
                                    = checkpointer.load_latest(return_path=True)
    starting_epoch = prev_epoch + 1
    
    seed_data, seed_hmm = jr.split(jr.PRNGKey(args.seed))
    
    dataset, train_dl, test_dl = \
        setup_data(seed_data, args.batch_size, args.seq_length,
                   (args.train, args.test), starting_epoch,
                   args.debug_max_files)
    
    if hmm is None:
        prior_kwargs = dict(
            emission_prior_scale= 1e-3, # default: 1e-4,
            emission_prior_extra_df= 0.5, # default: 0.,
        )

        tic = time.time()

        # In kmeans, default step_size=1200 corresponds to subsampling at 1 frame/min
        # FUTURE For k-means, consider bootstrapping covariance values (as opposed to using identity covariance).
        hmm = initialize_gaussian_hmm(args.hmm_init_method,
                                      seed_hmm, args.states, dataset.dim,
                                      dataloader=train_dl, step_size=1200,
                                      **prior_kwargs)
        toc = time.time()
        print(f"Initialized GaussianHMM using {args.hmm_init_method} init with {args.states} states. Elapsed time: {toc-tic:.1f}s\n")

    else:
        # TODO Make sure "warm-start" takes prior values into account
        print(f"Warm-starting from {warm_ckp_path}, training from epoch {starting_epoch}...\n")

    # ==========================================================================
    # Run
    fn = train_and_checkpoint
    fn_args = (train_dl, hmm, args.epochs, checkpointer)
    fn_kwargs = {'starting_epoch': starting_epoch,
                 'prev_train_lps': prev_train_lps,
                 'prev_test_lps': prev_test_lps,
                 'test_dataloader': test_dl,
                 'verbose': True}
    
    if args.mprof:
        mem_usage, (lps, last_ckp_path) = memory_usage(
                proc=(fn, fn_args, fn_kwargs), retval=True,
                backend='psutil_pss',
                stream=False, timestamps=True, max_usage=False,
                include_children=True, multiprocess=True,
        )

        # Save memory profiler results
        f_mprof = os.path.join(log_dir, 'train_and_checkpoint.mprof')
        write_mprof(f_mprof, mem_usage)
    else:
        lps, last_ckp_path = fn(*fn_args, **fn_kwargs)

    if test_lps is None:
        test_lp = compute_exact_lp(hmm, test_dl) / test_dl.total_emissions
        print("\n Final test_lp: ", test_lp)

    print (f"\nTraining completed, latest checkpoint saved at {last_ckp_path}")

    return

if __name__ == '__main__':
    main()