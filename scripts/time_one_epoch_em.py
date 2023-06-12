"""Compare per epoch run time of full-batch vs. stochastic EM algorithms.

The following three algorithms are compared:
    - `streaming_fullbatch_em_step`
    - `nonparallel_stochastic_em_step` -- (defined in `gaussian_hmm/_algorithms.py`)
    - `parallel_stochastic_em_step`    -- (defined in `gaussian_hmm/_algorithms.py`)

These step functions REDEFINED HERE to incorporate time-tracking commands
(but not the underlying E and M-steps).

Script is similar to `kf/fit_parallel_stem.py`, but with minor changes:
    - Arguments: split `seed` into `seed_init` and `seed_data`
    - Arguments: replaced `parallelize` with `method`
    - Arguments: Renamed `checkpoint_every` to `check_every`
    - Arguments: added `datadir` and `outdir`
    - `initialize_hmm`: initialize `prior_params` here, instead of in main()
"""

from cProfile import Profile
import pstats
import io

import os
from pathlib import Path
import argparse
from datetime import datetime
import h5py

import jax
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax


from kf.data import create_dataset_and_iterator, filter_min_frames
import kf.gaussian_hmm as GaussianHMM
from kf.inference import stochastic_train

# -------------------------------------
# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# ------------------------------------------------------------------------------

def select_files(datadir, seq_length, max_files=-1, seed_max_files=None):
    filepaths = sorted(datadir.glob('*.h5'))

    # If no .h5 files found, search recursively
    if len(filepaths) > 0:
        print(f'Found {len(filepaths)} files.')
    else:
        print(f'No *.h5 files found in {datadir}, searching recursively...', end="")
        filepaths = sorted(datadir.rglob('*.h5'))
        print(f'Found {len(filepaths)} files.')
    
    assert len(filepaths) > 0, f'Expected to find *.h5 files in {datadir}, no files found.'

    # Remove files that do not have enough minumum frames
    filepaths, _ = filter_min_frames(filepaths, seq_length)

    # If limiting the number of max files
    if max_files > 0 and (len(filepaths) > max_files):
        print(f"!!! WARNING !!! Limiting total number of files loaded to {max_files}.\n")
        idxs = jr.permutation(seed_max_files, len(filepaths))[:max_files]
        filepaths = [filepaths[i] for i in idxs]

    print(f'Loaded {len(filepaths)} files.')

    return filepaths


def initialize_hmm(seed, method, num_states, train_ds,
                   prior_scale, prior_extra_df,
                   subsample_step_size=1_200, verbose=True):
    """
    Arguments
        seed (jr.PRNGKey)
        method (str): Specifies HMM initialization method; either 'kmeans' or 'random'.
        num_states (int): Number of states to initialize HMM with
        train_ds (MultiSessionDataset):
        prior_scale (Scalar):
        prior_extra_df (Scalar):
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
                subsample_step_size)

            with h5py.File(ds.filepath, 'r') as f:
                subsampled_dataset.append(
                    jnp.asarray(f['stage/block0_values'][subsampled_slice]))
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
    
    # Initialize rolling statistics
    init_stats = GaussianHMM.initialize_statistics(num_states, emissions_dim)
    
    # Set GaussianHMM prior parameters to non-informative values, except
    # boost the prior parameters associated with emission covariance matrices
    # (i.e. emission_scale and emission_extra_df) to scale of minibatch size
    # to regularize covariance matrix to be PSD
    prior_params = GaussianHMM.initialize_prior_from_scalar_values(
        num_states,
        emissions_dim,
        emission_scale=prior_scale,
        emission_extra_df=prior_extra_df,)
    
    return init_params, init_stats, prior_params

# ==============================================================================

def main(args):
    
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    algorithm = args.algorithm
    parallelize = True if algorithm == 'pstochastic' else False

    # Set user-specified seed
    seed_init = jr.PRNGKey(args.seed_init)
    seed_data = jr.PRNGKey(args.seed_data)
    print(f"Initializing model with seed {args.seed_init}")
    print(f"Selecting data with seed {args.seed_data}")

    # Algorithm parameters
    num_states = args.states
    init_method = args.hmm_init_method
    subsample_step_size = args.hmm_init_subsample
    seq_length = args.seq_length
    local_batch_size = args.batch_size_per_device
    n_epochs = args.epochs

    # Prior parameters
    prior_scale = args.prior_scale
    prior_extra_df = args.prior_extra_df

    # Schedule parameters
    schedule_decay = args.schedule_decay

    # Set session name
    # if args.session_prefix is None:
    #     timestamp = datetime.now().strftime("%y%m%d_%H%M")
    #     session_prefix = f'{timestamp}'
    # else:
    #     session_prefix = args.session_prefix
    # _str_debug = args.debug_max_files if args.debug_max_files > 0 else 'all'
    
    # log_dir = os.path.join(outdir, session_name)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # print(f"Output files will be logged to: {log_dir}\n")

    # Checkpointing parameters
    # check_every = args.check_every
    # checkpoint_dir = log_dir
    # n_checkpoints_to_keep = args.checkpoints_to_keep

    # Print JAX precision
    dtype = jnp.array([1.2, 3.4]).dtype
    print(f"JAX using {dtype} precision\n")
    # ==========================================================================
    
    # Set-up training dataset
    seed_max_files, seed_dataset = jr.split(seed_data)
    filepaths = select_files(datadir, seq_length, args.debug_max_files, seed_max_files)
    train_ds, train_dl = create_dataset_and_iterator(
            seed_dataset, filepaths, seq_length, local_batch_size, parallelize,)
    emissions_dim = train_ds.sequence_shape[-1]

    total_emissions_per_batch = train_dl.dataset.num_samples // len(train_dl)
    print(f'total emissions per batch {total_emissions_per_batch:.1e}')

    # Initialize GaussianHMM parameters via 'random' or 'kmeans'
    # Set GaussianHMM prior parameters to non-informative values, except
    # boost the prior parameters associated with emission covariance matrices
    # (i.e. emission_scale and emission_extra_df) to scale of minibatch size
    # to regularize covariance matrix to be PSD
    init_params, init_stats, prior_params = initialize_hmm(
        seed_init, init_method, num_states, train_ds,
        prior_scale*total_emissions_per_batch,
        prior_extra_df*total_emissions_per_batch,
        subsample_step_size=subsample_step_size, verbose=True,
    )

    print(f'prior scale {prior_params.emission_scale[0,0,0]:.1e}, ' \
          + f'dim+extra_df {prior_params.emission_df[0]:.1e}')

    if algorithm == 'fullbatch':
        # def init_fn():
        #     return prior_params, init_params, init_stats, []

        def run_one_epoch(params, stats):
            params, _, _ = GaussianHMM.streaming_em_step(prior_params, params, stats, train_dl)
            params.emission_covariances.block_until_ready()
            return

    else:
        schedule = optax.exponential_decay(
            init_value=1.,
            end_value=0.,
            transition_steps=len(train_dl),
            decay_rate=schedule_decay,
        )
        learning_rates = schedule(jnp.arange(n_epochs * len(train_dl)))
        learning_rates = learning_rates.reshape(n_epochs, len(train_dl))

        # def init_fn():
        #     return prior_params, init_params, init_stats, [], train_dl.batch_sampler.state
        
        # Parallel stochastic EM step can NOT be broken down for timing.
        if algorithm == 'pstochastic':
            step_fn = GaussianHMM.parallel_stochastic_em_step
        elif algorithm == 'stochastic':
            step_fn = GaussianHMM.nonparallel_stochastic_em_step

        def run_one_epoch(params, rolling_stats,):
            for minibatch, minibatch_emissions in enumerate(train_dl):
                params, rolling_stats, _ = step_fn(
                    prior_params, params, rolling_stats,
                    learning_rates[0][minibatch], minibatch_emissions)
            params.emission_covariances.block_until_ready()
            return
                
    prof = Profile()
    prof.runcall(run_one_epoch, init_params, init_stats)

    # Print stats to an IO stream, then save
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s)
    ps.sort_stats('cumtime')
    ps.print_stats()
    
    n_batches = len(train_dl) * train_dl.batch_sampler.batch_size
    session_name = f'{algorithm}-{n_batches}_batches-{args.seed_data}_seed_data'
    with open(outdir/f'{session_name}.prof', 'w+') as f:
        f.write(s.getvalue())
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time-profile full-batch vs. stochastic EM algorithms.')
    parser.add_argument(
        '--datadir', type=str,
        help='Path to directory containing eigenpose data.')
    parser.add_argument(
        '--outdir', type=str,
        help='Path to parent output directory. Files ulimately stored in `outdir/SESSION_PREFIX-*/`.')
    
    parser.add_argument(
        '--algorithm', type=str,
        choices=['fullbatch', 'stochastic', 'pstochastic'],
        help='EM algorithm to use. If using `pstochastic`, must set XLA_FLAGS to expose number of cores.'
    )

    parser.add_argument(
        '--session_prefix', type=str, default=None,
        help='Identifying token. Used for log and checkpoint files')
    parser.add_argument(
        '--seed_init', type=int, required=True,
        help='RNG seed for intializing HMM.')
    parser.add_argument(
        '--seed_data', type=int, required=True,
        help='RNG seed for selecting and splitting data.')
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
        '--check_every', type=int, default=50,
        help='Number of iterations between which to checkpoint'
    )
    parser.add_argument(
        '--checkpoints_to_keep', type=int, default=0,
        help='Number of checkpoints to keep. If 0, keep all.'
    )

    parser.add_argument(
        '--debug_max_files', type=int, default=-1,
        help='FOR DEBUGGING: Maximum number of files (~days of recording) in directory to expose. Default: -1, expose all.')

    args = parser.parse_args()

    main(args)