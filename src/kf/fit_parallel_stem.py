"""Script for fitting HMM to kf data via stochastic EM.

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
import h5py

import jax
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax

from torch.utils.data import DataLoader
from kf.data import (MultiSessionDataset,
                     RandomBatchSampler,
                     filter_min_frames)
import kf.gaussian_hmm as GaussianHMM
from kf.inference import fit_stochastic_em

DATADIR = Path(os.environ['DATADIR'])
TEMPDIR = Path(os.environ['TEMPDIR'])

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
    '--hmm_init_subsample', type=int, default=600,
    help='1 sample / N frames, to create subsampled dataset for k-means initialization.')
parser.add_argument(
    '--batch_size_per_device', type=int, default=1,
    help='Number of batches loaded per device per iteration.')
parser.add_argument(
    '--seq_length', type=int, default=72000,
    help='Number of consecutive frames per sequence.')
parser.add_argument(
    '--epochs', type=int, default=10,
    help='Number of stochastic EM iterations to run')
parser.add_argument(
    '--states', type=int, default=20,
    help='Number of HMM states to fit')

parser.add_argument(
    '--prior_scale', type=float, default=0.0001,
    help='Scale of prior NIW distribution')
parser.add_argument(
    '--prior_extra_df', type=float, default=0.1,
    help='Extra DOF of prior NIW distribution')

parser.add_argument(
    '--schedule_decay', type=float, default=0.95,
    help='Decay rate of exponential learning schedule for stochastic EM annealing.'
)

parser.add_argument(
    '--checkpoint_every', type=int, default=50,
    help='Number of iterations between which to checkpoint'
)
parser.add_argument(
    '--checkpoints_to_keep', type=int, default=0,
    help='Number of checkpoints to keep. If 0, keep all.'
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

def random_split(key, elements, sizes):
    """Split sequence of elemtns into subsets of specified sizes."""
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
    
def initialize_training_data(
        seed,
        em_seq_length,
        em_local_batch_size,
        em_parallelize,
        debug_max_files=-1,
        verbose=True,
    ):
    """Initialize dataset and dataloader for training via stochastic EM.

    Arguments
        seed (jr.PRNGKey)
        em_seq_length (int): Number of frames in a sequence, for training.
        em_local_batch_size(int): Number of sequences to load per minibatch, for training.
        em_parallelize (bool): If True, using parallel algorithm
        debug_max_files (int): Maximum number of files to use in the dataset.
            Default: -1, use all files found in data directory.
        verbose (bool): If True, print status and shape messages.
    """

    seed_debug, seed_slice, seed_batch = jr.split(seed, 3)
    # Print Jax precision
    dtype = jnp.array([1.2, 3.4]).dtype
    if verbose: print(f"JAX using {dtype} precision\n")

    # =====================
    # Initialize filepaths
    # =====================
    filepaths = sorted(DATADIR.glob('*.h5'))

    # If no .h5 files found, search recursively
    if len(filepaths) > 0:
        print(f'Found {len(filepaths)} files.')
    else:
        print(f'No *.h5 files found in {DATADIR}, searching recursively...', end="")
        filepaths = sorted(DATADIR.rglob('*.h5'))
        print(f'Found {len(filepaths)} files.')
    
    assert len(filepaths) > 0, f'Expected to find *.h5 files in {DATADIR}, no files found.'

    # Remove files that do not have enough minumum frames
    filepaths, _ = filter_min_frames(filepaths, em_seq_length)

    if debug_max_files > 0:
        print(f"!!! WARNING !!! Limiting total number of files loaded to {debug_max_files}.\n")
        idxs = jr.permutation(seed_debug, len(filepaths))[:debug_max_files]
        filepaths = [filepaths[i] for i in idxs]

    if verbose:
        print(f'Loaded {len(filepaths)} files.')

    # ============================
    # Initialize training dataset
    # ============================
    train_ds = MultiSessionDataset(filepaths, seed_slice, em_seq_length, dtype=dtype)
    emissions_dim = train_ds.sequence_shape[-1]

    # Define how to load a batch of emissions gets reshaped
    num_devices = jax.local_device_count()
    em_batch_size = em_local_batch_size * num_devices
    if em_parallelize:
        assert num_devices > 1, f'Expected >1 device to parallelize, only see {num_devices}. Double-check XLA_FLAGS setting.'
        # local_batch_size = batch_size // num_devices
        def collate_fn(sequences):
            return onp.stack(sequences, axis=0).reshape(num_devices, em_local_batch_size, *sequences[0].shape)
        
    else:
        assert num_devices == 1, f'Expected 1 device, but seeing {num_devices}. Reset XLA_FLAGS setting.'
        def collate_fn(sequences):
            return onp.stack(sequences, axis=0)

    # Construct dataloader for EM fitting
    sampler = RandomBatchSampler(train_ds, em_batch_size, seed_batch)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    if verbose: 
        print(f'num_devices {num_devices}, local_batch_size {em_local_batch_size}, batch_size {em_batch_size}') 
        print("Initialized training dataset...")
        print(f"\t{len(train_ds):3d} sets of {em_seq_length/72000:.1f} hr sequences")
        print(f"\t{len(train_dl):3d} batches of {em_batch_size} sequences")
        print()

    return train_ds, train_dl

def initialize_hmm(seed, method, num_states, train_ds,
                   subsample_step_size=1_200, verbose=True):
    """
    Arguments
        seed (jr.PRNGKey)
        method (str): Specifies HMM initialization method; either 'kmeans' or 'random'.
        num_states (int): Number of states to initialize HMM with
        train_ds (MultiSessionDataset):
        subsample_step_size (int): Number of frames to subsample training data.
            Used to prepare data for k-means. Choose value to be large enough
            that we get a meaningful reduction in data, but not so small that
            kmeans takes too long. Default: 1200, corresponding to 1 fr/min @ 20 Hz.
    
    Returns
        init_params (GaussianHMM.Parameters)
        subsampled_dataset (list or ndarray): If method=='kmeans', array with
            shape (n,d) subsampled from training dataset. If method=='random',
            empty list. Convenience function for re-evaluating k-means inits.
    """
    emissions_dim = train_ds.sequence_shape[-1]
    
    # Subsample from training dataset manually, if using k-means
    subsampled_dataset = []
    if method == 'kmeans':
        for ds in train_ds.datasets:
            subsampled_slice = slice(
                ds.sequence_slices[0].start,
                ds.sequence_slices[-1].stop,
                subsample_step_size
                )

            with h5py.File(ds.filepath, 'r') as f:
                subsampled_dataset.append(
                    jnp.asarray(f['stage/block0_values'][subsampled_slice])
                )
        
        subsampled_dataset = jnp.concatenate(subsampled_dataset)

        if verbose:
            num_subsampled = len(subsampled_dataset)
            num_total = train_ds.num_samples
            print(f'Fitting k-means with {num_subsampled}/{num_total} frames, ' + \
                f'{num_subsampled/num_total*100:.2f}% of training data...' + \
                f'Subsampled at {subsample_step_size / 60 / 20:.2f} frames / min.')
    
    init_params = GaussianHMM.initialize_model(
        seed, method, num_states, emissions_dim, subsampled_dataset,
        emission_covs_scale=1., emission_covs_method='bootstrap')
    
    return init_params, subsampled_dataset

# -------------------------------------

def main():
    args = parser.parse_args()

    # Set user-specified seed
    seed = jr.PRNGKey(args.seed)
    seed_data, seed_init = jr.split(seed, 2)
    print(f"Setting user-specified seed: {args.seed}")

    # Algorithm parameters
    num_states = args.states
    init_method = args.hmm_init_method
    subsample_step_size = args.hmm_init_subsample
    seq_length = args.seq_length
    local_batch_size = args.batch_size_per_device

    num_epochs = args.epochs
    parallelize = args.parallelize

    # Prior parameters
    prior_scale = args.prior_scale
    prior_extra_df = args.prior_extra_df

    # Schedule parameters
    schedule_decay = args.schedule_decay

    # Set session name
    if args.session_prefix is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        session_prefix = f'{timestamp}'
    else:
        session_prefix = args.session_prefix
    _str_debug = args.debug_max_files if args.debug_max_files > 0 else 'all'
    _str_parallel = '-parallel' if parallelize else ''
    session_name = f'{session_prefix}-{num_states}_states-{init_method}_init-{_str_debug}_files{_str_parallel}'
    log_dir = os.path.join(TEMPDIR, session_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Output files will be logged to: {log_dir}\n")

    # Checkpointing parameters
    checkpoint_every = args.checkpoint_every
    checkpoint_dir = log_dir
    num_checkpoints_to_keep = args.checkpoints_to_keep
    # ==========================================================================
        
    # Set-up training dataset
    train_ds, train_dl = initialize_training_data(
                            seed_data,
                            seq_length, local_batch_size, parallelize,
                            debug_max_files=args.debug_max_files)
    emissions_dim = train_ds.sequence_shape[-1]

    # Initialize GaussianHMM parameters via 'random' or 'kmeans'
    init_params, _ = initialize_hmm(
        seed_init, init_method, num_states, train_ds,
        subsample_step_size=subsample_step_size, verbose=True,
    )

    # Set GaussianHMM prior parameters to non-informative values, except
    # boost the prior parameters associated with emission covariance matrices
    # (i.e. emission_scale and emission_extra_df) to scale of minibatch size
    # to regularize covariance matrix to be PSD
    total_emissions_per_batch = train_dl.dataset.num_samples // len(train_dl)
    prior_params = GaussianHMM.initialize_prior_from_scalar_values(
        num_states,
        emissions_dim,
        emission_scale=prior_scale*total_emissions_per_batch,
        emission_extra_df=prior_extra_df*total_emissions_per_batch,)
    
    print(f'total emissions per batch {total_emissions_per_batch:.1e}')
    print(f'prior scale {prior_scale*total_emissions_per_batch:.1e}, extra_df {prior_extra_df*total_emissions_per_batch:.1e}')
    
    # If num_checkpoints_to_keep is not positive int, keep all checkpoints
    if num_checkpoints_to_keep < 1:
        num_checkpoints_to_keep = num_epochs * len(train_dl) + 1

    # Run function
    schedule = optax.exponential_decay(
            init_value=1.,
            end_value=0.,
            transition_steps=len(train_dl),
            decay_rate=schedule_decay,
        )

    fn_kwargs = {
        'num_epochs': num_epochs,
        'parallelize': parallelize,
        'checkpoint_every': checkpoint_every,
        'checkpoint_dir': checkpoint_dir,
        'num_checkpoints_to_keep': num_checkpoints_to_keep,
        'schedule': schedule,
    }
    fitted_params, lps = fit_stochastic_em(init_params, prior_params, train_dl, **fn_kwargs)
    
    print(lps)

    return fitted_params, lps

if __name__ == '__main__':
    main()
