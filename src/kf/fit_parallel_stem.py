"""Script for profiling CPU memory usage during stochastic EM fit of HMM to kf data.

Flags to set in environment
---------------------------
REQUIRED
    DATADIR - Folder path to where data folder (specified by `fish_id`) resides
    TEMPDIR - Folder path to store output and log files

OPTIONAL
    JAX_ENABLE_X64 - True or False. If True, all computations performed in x64.
    XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]
        - Number of CPUs to make visible to JAX. Set if pmap-ing
"""

import os
from pathlib import Path
import argparse
from datetime import datetime
from memory_profiler import memory_usage

import jax
import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from torch.utils.data import DataLoader
from kf.data import (SingleSessionDataset,
                     MultiSessionDataset,
                     RandomBatchSampler,
                     get_file_raw_shapes)
import kf.gaussian_hmm as GaussianHMM
from kf.inference import fit_stochastic_em

DATADIR = Path(os.environ['DATADIR'])
TEMPDIR = Path(os.environ['TEMPDIR'])
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

parser = argparse.ArgumentParser(description='Profile stochastic EM algorithms')
parser.add_argument(
    '--session_prefix', type=str, default=None,
    help='Identifying token,. Used for log and checkpoint files')
parser.add_argument(
    '--seed', type=int, required=True,
    help='Initial RNG seed, for splitting data and intializing HMM.')
parser.add_argument(
    '--parallelize', action='store_true',
    help='If specified, run parallel stochastic EM algorithm over multiple cores.')
parser.add_argument(
    '--hmm_init_method', type=str, default='kmeans',
    choices=['random', 'kmeans'],
    help='HMM initialization method in the first epoch.')
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
    '--checkpoint_every', type=int, default=50,
    help='Number of iterations between which to checkpoint'
)

parser.add_argument(
    '--debug_max_files', type=int, default=-1,
    help='FOR DEBUGGING: Maximum number of files (~days of recording) in directory to expose. Default: -1, expose all.')

# ------------------------------------------------------------------------------
def write_mprof(path: str, mem_usage: list, mode: str='w+') -> None:
    """Writes time-based memory profiler results to file."""
    with open(path, mode) as f:
        for res in mem_usage:
            f.writelines('MEM {} {}\n'.format(res[0], res[1]))
    return

def setup_dataset(seed, train_test_size, debug_max_files, seq_length, dtype):
    """Setup dataset object, get sequence slices"""

    seed_slice, seed_debug, seed_split = jr.split(seed, 3)

    # TODO This is hard fixed to single subject for now
    filepaths = sorted((DATADIR/fish_id).glob('*.h5'))

    if debug_max_files > 0:
        print(f"!!! WARNING !!! Limiting total number of files loaded to {debug_max_files}.")
        idxs = jr.permutation(seed_debug, len(filepaths))[:debug_max_files]
        filepaths = [filepaths[i] for i in idxs]

    # Split filepaths into train and test sets
    def random_split(key, elements, sizes):
        N = len(elements)

        # Fractional sizes are given, convert into number of samples (int)
        if all([(sz <= 1 for sz in sizes)]):
            subset_sizes = [int(frac * N) for frac in sizes]

            # If input sizes sum up to 1, make sure to distribute all samples
            if jnp.isclose(sum(sizes), 1) and (sum(subset_sizes) < N):
                remainder = N - sum(subset_sizes)
                for i in range(remainder):
                    subset_sizes[i % len(sizes)] += 1
            
            sizes = subset_sizes

        # Return list of lists: outer size has len(sizes), inner size according to 
        indices = jr.permutation(key, N)[:sum(sizes)]
        return [
            [elements[i] for i in indices[(offset-size):offset]]
            for offset, size in zip(onp.cumsum(sizes), sizes)
        ]
    train_filepaths, test_filepaths = random_split(seed_split, filepaths, train_test_size)
    
    # Construct train and test dataset objects
    train_seq_key, test_seq_key = jr.split(seed_slice)
    train_ds = MultiSessionDataset(train_filepaths, train_seq_key, seq_length, dtype=dtype)
    test_ds = MultiSessionDataset(test_filepaths, test_seq_key, seq_length, dtype=dtype)
    return train_ds, test_ds

# -------------------------------------

def main():
    args = parser.parse_args()

    # Set user-specified seed
    seed = jr.PRNGKey(args.seed)
    seed_data, seed_dl, seed_init = jr.split(seed, 3)
    print(f"Setting user-specified seed: {args.seed}")

    # Print Jax precision
    dtype = jnp.array([1.2, 3.4]).dtype
    print(f"JAX using {dtype} precision\n")

    # Algorithm parameters
    num_states = args.states
    init_method = args.hmm_init_method

    num_epochs = args.epochs

    parallelize = args.parallelize

    # Set session name
    if args.session_prefix is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        session_prefix = f'{timestamp}'
    else:
        session_prefix = args.session_prefix
    _str_debug = args.debug_max_files if args.debug_max_files > 0 else 'all'
    _str_parallel = '-parallel' if args.parallelize else ''
    session_name = f'{session_prefix}-{num_states}_states-{init_method}_init-{_str_debug}_files-{init_method}_init{_str_parallel}'
    log_dir = os.path.join(TEMPDIR, session_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Output files will be logged to: {log_dir}\n")
    
    # ==========================================================================
    
    # Setup dataset
    batch_size = args.batch_size
    train_test_size = (args.train, args.test)
    seq_length = args.seq_length

    train_ds, test_ds = setup_dataset(seed_data, train_test_size, args.debug_max_files, seq_length, dtype)
    emission_dim = train_ds.sequence_shape[-1]

    # Define how a batch of emissions gets reshaped, depending on if using parallelization
    num_devices = jax.local_device_count()
    if parallelize:
        assert num_devices > 1, f'Expected >1 device to parallelize, only see {num_devices}. Double-check XLA_FLAGS setting.'
        local_batch_size = batch_size // num_devices
        def collate_fn(sequences):
            return onp.stack(sequences, axis=0).reshape(num_devices, local_batch_size, *sequences[0].shape)
        
    else:
        assert num_devices == 1, f'Expected 1 device, but seeing {num_devices}. Reset XLA_FLAGS setting.'
        def collate_fn(sequences):
            return onp.stack(sequences, axis=0)

    # Construct dataloader
    sampler = RandomBatchSampler(train_ds, batch_size, seed_dl)
    train_dataloader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)
    
    print("Initialized training dataset with "
            + f"{len(train_ds):3d} sets of {seq_length/72000:.1f} hr sequences; "
            + f"{len(train_dataloader):3d} batches of {batch_size} sequences per batch")
    print()

    # Initialize GaussianHMM model parameters based on specified method
    init_params = GaussianHMM.initialize_model(init_method, seed_init, num_states, emission_dim, train_dataloader)
    
    # Set GaussianHMM prior parameters to non-informative values, except
    # boost the prior parameters associated with emission covariance matrices
    # (i.e. emission_scale and emission_extra_df) to scale of minibatch size
    # to regularize covariance matrix to be PSD
    total_emissions_per_batch = train_dataloader.dataset.num_samples // len(train_dataloader)
    prior_params = GaussianHMM.initialize_prior_from_scalar_values(
        num_states,
        emission_dim,
        emission_scale=(1e-4)*total_emissions_per_batch,
        emission_extra_df=(1e-1)*total_emissions_per_batch,)
    
    # Run function
    fn_kwargs = {
        'num_epochs': num_epochs,
        'parallelize': parallelize,
        'checkpoint_every': args.checkpoint_every,
        'checkpoint_dir': log_dir,
        'num_checkpoints_to_keep': 10
    }
    fitted_params, lps = fit_stochastic_em(init_params, prior_params, train_dataloader, **fn_kwargs)

    print(lps)

    return 

if __name__ == '__main__':
    main()