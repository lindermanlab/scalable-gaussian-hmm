# Script for profiling HMM fit via pmap parallelization

import os
import argparse
from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.models import GaussianHMM
from kf.data_utils import FishPCDataset, FishPCLoader, save_hmm

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

parser = argparse.ArgumentParser(description='Single-subject EM parallelization')
parser.add_argument(
    '--method', type=str, default='pmap',
    choices=['pmap','vmap'],
    help='Parallelization approach. NB: If using pmap, must run on node with multiple cores.')
parser.add_argument(
    '--log_dir', type=str, default=TEMPDIR,
    help='Directory to log profiles.')
parser.add_argument(
    '--log_prefix', type=str, default=None,
    help='Prefix for log files.')
parser.add_argument(
    '--seed', type=int, default=45212276,
    help='PRNG seed to split data and intialize HMM.')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Number of batches loaded per iteration.')
parser.add_argument(
    '--frames_per_batch', type=int, default=100000,
    help='Number of frames per batch. Also used to filter out files which do not have this many frames.')
parser.add_argument(
    '--max_frames_per_day', type=int, default=-1,
    help='Maximum number of frames available in each file/day. Useful for debugging.')
parser.add_argument(
    '--train', type=float, default=1,
    help='If >1, number of files in dataset to train over. If [0, 1), fraction of dataset to train over.')
parser.add_argument(
    '--test', type=float, default=1,
    help='If >1, number of files in dataset to test over. If [0, 1), fraction of dataset to test over.')
parser.add_argument(
    '--iters', type=int, default=10,
    help='Number of EM iterations to run')
parser.add_argument(
    '--states', type=int, default=20,
    help='Number of HMM states to fit')
parser.add_argument(
    '--mprof', action='store_true',
    help='If specified, profile memory usage with memory_profiler.')
parser.add_argument(
    '--savefit', action='store_true',
    help='If specified, save fitted HMM.')

def write_mprof(path: str, mem_usage: list, mode: str='w+') -> None:
    """Writes time-based memory profiler results to file."""

    with open(path, mode) as f:
        for res in mem_usage:
            f.writelines('MEM {} {}\n'.format(res[0], res[1]))
    return
    
# =============================================================================
def main():
    # TODO Allow warm-starting of hmm fit code from saved file.
    # - Add argument to to start with random initialization or from file
    # - If warm-starting, extract last iteration and log likelihoods, same seed for splitting dataset
    args = parser.parse_args()
    method = args.method
    log_dir = args.log_dir
    log_prefix = args.log_prefix        

    seed = jr.PRNGKey(args.seed)
    batch_size = args.batch_size
    frames_per_batch = args.frames_per_batch
    max_frames_per_day = args.max_frames_per_day

    num_train = args.train
    num_test = args.test
    
    num_hmm_states = args.states
    num_em_iters = args.iters

    # TODO Add these options into arg parser
    mprof = args.mprof
    save_results = args.savefit

    if log_prefix is None:
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        log_prefix=f"{timestamp}-{num_hmm_states}st-{num_em_iters}it"
    
    print("")
    print(f"Output files will be logged to: {log_dir}")
    # ==========================================================================
    seed_train, seed_test, seed_init_hmm = jr.split(seed, 3)

    fish_dir = os.path.join(DATADIR, fish_id)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])
    
    num_train = int(num_train) if num_train >= 1 else int(len(filepaths) * num_train)
    num_test = int(num_test) if num_test >= 1 else int(len(filepaths) * num_test)
    
    train_filepaths = filepaths[:num_train]
    test_filepaths = filepaths[num_train:num_train+num_test]

    # Define training and test datasets and data loaders
    train_ds = FishPCDataset(train_filepaths)
    test_ds = FishPCDataset(test_filepaths)

    # We don't care about labels right now
    collate_fn = lambda seq_label_pairs: tuple(map(jnp.stack, zip(*seq_label_pairs)))[0]
    train_dl = FishPCLoader(train_ds,
                            seq_length=frames_per_batch,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=True,
                            seed=int(seed_train[-1]))
    # total_num_train = len(train_dl) * batch_size * frames_per_batch
    total_num_train = train_dl.total_emissions

    test_dl = FishPCLoader(test_ds,
                           seq_length=frames_per_batch,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=True,
                           seed=int(seed_test[-1]))

    print(f"Initialized datasets ({num_train} days training, {num_test} days testing)")
    print(f"\ttrain: {len(train_dl):3d} batches of shape {(batch_size, frames_per_batch, train_ds.dim)}")
    print(f"\ttest : {len(test_dl):3d} batches of shape {(batch_size, frames_per_batch, test_ds.dim)}")
        
    # --------------------------------------------------------------------------
    # Initialize hidden Markov model
    print(f'\nInitializing HMM with {num_hmm_states} states...\n')
    hmm = GaussianHMM.random_initialization(seed_init_hmm, num_hmm_states, train_ds.dim)

    # Run function
    fn_args = (train_dl, total_num_train)
    fn_kwargs = {'num_epochs': num_em_iters}

    if mprof:
        from memory_profiler import memory_usage
        
        # In order to record memory usage when calling `memory_usage` on a
        # python function, MUST specify stream=False (otherwise, returns None)
        # and then manually write results to file
        # NB: `memory_usage` only automatically writes to file if called by an
        #   external process, i.e. with the mprof command in the command line.
        #   When `mprof` called via a job, psutil has trouble finding the
        #   correct process id and throws a NoProcessFound error. It still seems
        #   to be able record it when submitted as non-interactive job, but
        #   program fails when submitted interactively
        mem_usage, (train_lls) = memory_usage(
                proc=(hmm.fit_stochastic_em, fn_args, fn_kwargs), retval=True,
                backend='psutil_pss',
                stream=False, timestamps=True, max_usage=False,
                include_children=True, multiprocess=True,
        )

        f_mprof = os.path.join(log_dir, log_prefix+'.mprof')
        write_mprof(f_mprof, mem_usage)
    else:
        train_lls = hmm.fit_stochastic_em(*fn_args, **fn_kwargs)
        train_lls.block_until_ready()

    print('expected_train_lls:')
    num_train_emissions = len(train_dl) / batch_size / frames_per_batch
    train_lls /= num_train_emissions
    for epoch, ll in enumerate(train_lls):
        print(f"{epoch:2d}: {ll:.4f}")
    
    # Evaluate on test data
    print("Evaluating test log likelihood")
    test_ll = 0.
    num_test_emissions = len(test_dl) / batch_size / frames_per_batch
    for batch_emissions in test_dl:
        batch_stats = hmm.e_step(batch_emissions)
        test_ll += batch_stats.marginal_loglik.sum()
    test_ll /= num_test_emissions
    print(f'average_test_ll: {test_ll:.4f}')
    
    # --------------------------------------------------------------------------
    # Save likelihoods and hmm
    if save_results:
        fpath = os.path.join(log_dir, log_prefix+'.npz')
        save_hmm(fpath, hmm, train_lls=train_lls, test_lls=test_ll)
    return

if __name__ == '__main__':
    main()