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

import jax.numpy as jnp
import jax.random as jr

from ssm_jax.hmm.models import GaussianHMM
from kf import (FishPCDataset, FishPCLoader,
                kmeans_initialization,
                CheckpointDataclass, train_and_checkpoint,)

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

def setup_data(seed, batch_size, seq_length, split_sizes, starting_epoch=0, DEBUG_MAX_FILES=-1):
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
    seed_split, seed_dl = jr.split(seed,)
    seed_dl_train = jr.fold_in(seed_dl, starting_epoch)

    # TODO This is hard fixed to single subject for now
    fish_dir = os.path.join(DATADIR, fish_id)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])

    # TODO Hard limiting number of filepaths. Remove when done debugging
    if DEBUG_MAX_FILES > 0:
        print(f"!!! WARNING !!! Limiting total number of files loaded to {DEBUG_MAX_FILES}.")
    filepaths = filepaths[:DEBUG_MAX_FILES]
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
          + f"{len(train_dl):3d} batches of {batch_size} sequences")
    print("Initialized testing  dataset with "
          + f"{len(test_slices):3d} sets of {seq_length/72000:.1f} hr sequences; "
          + f"{len(test_dl):3d} batches of {batch_size} sequences")
    print()

    return dataset, train_dl, test_dl

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
    hmm, prev_epoch, prev_lps, warm_ckp_path = checkpointer.load_latest(return_path=True)
    starting_epoch = prev_epoch + 1
    
    seed_data, seed_hmm = jr.split(jr.PRNGKey(args.seed))
    
    dataset, train_dl, test_dl = \
        setup_data(seed_data, args.batch_size, args.seq_length,
                   (args.train, args.test),
                   starting_epoch,
                   args.debug_max_files)
    
    if hmm is None:
        print(f'Initializing HMM with {args.states} states...\n')
        if args.hmm_init_method == 'random':
            hmm = GaussianHMM.random_initialization(seed_hmm, args.states, dataset.dim)
        elif args.hmm_init_method == 'kmeans':
            # TODO Parameterize subset_size
            print("!!! WARNING !!! kmeans initialization specified -- currently slow and not optimized.")
            hmm = kmeans_initialization(seed_hmm, args.states, dataset, subset_size=144000)
    else:
        print(f"Warm-starting from {warm_ckp_path}, training from epoch {starting_epoch}...\n")

    # Run
    fn = train_and_checkpoint
    fn_args = (train_dl, hmm, args.epochs, checkpointer)
    fn_kwargs = {'starting_epoch': starting_epoch, 'prev_lps': prev_lps}
    if args.mprof:        
        mem_usage, (train_lps, last_ckp_path) = memory_usage(
                proc=(fn, fn_args, fn_kwargs), retval=True,
                backend='psutil_pss',
                stream=False, timestamps=True, max_usage=False,
                include_children=True, multiprocess=True,
        )

        # train_lps.block_until_ready()

        f_mprof = os.path.join(log_dir, 'train_and_checkpoint.mprof')
        write_mprof(f_mprof, mem_usage)
    else:
        train_lps, last_ckp_path = fn(*fn_args, **fn_kwargs)
        # train_lps.block_until_ready()

    print('expected_train_lls:')
    train_lps /= train_dl.total_emissions
    for epoch, lp in enumerate(train_lps):
        print(f"{epoch:2d}: {lp:.4f}")
    
    # ==========================================================================
    # Evaluate on test data
    print("Evaluating test log likelihood")
    test_lp = 0.
    for batch_emissions in test_dl:
        batch_stats = hmm.e_step(batch_emissions)
        test_lp += batch_stats.marginal_loglik.sum() / test_dl.total_emissions
    print(f'average_test_lp: {test_lp:.4f}')
    
    return

if __name__ == '__main__':
    main()