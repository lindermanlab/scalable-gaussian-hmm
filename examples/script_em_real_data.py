# Time EM on real data

import os

from jax import jit, vmap, pmap, lax
import jax.numpy as np
import jax.random as jr

import jax.profiler

from functools import partial, reduce

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step, collective_m_step,
                          fullbatch_e_step, fullbatch_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)

from kf.data_utils import FishPCDataset, FishPCDataloader

import pdb
from tqdm import tqdm
# -------------------------------------
# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -------------------------------------

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']

def vmap_em_step(hmm, split_emissions):
    # Since we vmap'd arrays directly, results in a single SuffStats
    # instance with batch shape (M,...). No addt'l transforms req'd.
    normd_suff_stats = vmap(partial(sharded_e_step, hmm))(split_emissions)
    
    new_hmm = collective_m_step(normd_suff_stats)

    return new_hmm, normd_suff_stats

def fit_hmm_nomap(train_dataset, test_dataset, hmm, num_iters):
    """Fit HMM to FishPCDataset using expectation maximization.

    Parameters
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        hmm: GaussianHMM
            Randomly initialized or warm-started hmm.
        num_iters: int
            Number of EM steps to run.
    """
    # @jit
    def _em_step(hmm, emissions):
        posterior = fullbatch_e_step(hmm, emissions)
        hmm = fullbatch_m_step(posterior, emissions)
        avg_ll = posterior.marginal_loglik / len(emissions)
        return hmm, avg_ll

    emissions_dim = train_dataset.dim

    # Load all emissions, shape (num_days * uniform_num_frames_per_day, dim)
    
    train_dl = FishPCDataloader(train_dataset, batch_size=len(train_dataset))
    train_emissions = next(iter(train_dl)).reshape(-1, emissions_dim)
    print(f'Loaded training data...shape {train_emissions.shape}')

    test_dl = FishPCDataloader(test_dataset, batch_size=len(test_dataset))
    test_emissions = next(iter(test_dl)).reshape(-1, emissions_dim)
    print(f'Loaded testing data...shape {test_emissions.shape}')

    train_lls = - np.ones(num_iters) * np.inf
    test_lls  = - np.ones(num_iters) * np.inf

    test_lls.block_until_ready()
    print('Beginning fit...')
    for itr in tqdm(range(num_iters)):
        # pdb.set_trace()
        hmm, avg_train_ll = _em_step(hmm, train_emissions)
        
        _, avg_test_ll = _em_step(hmm, test_emissions)

        train_lls = train_lls.at[itr].set(avg_train_ll)
        test_lls = test_lls.at[itr].set(avg_test_ll)

        jax.profiler.save_device_memory_profile(f"memory{itr}_{len(train_emissions)}.prof")
    return train_lls, test_lls

def fit_hmm_pmap(em_step, train_dataset, test_dataset, initial_hmm, seed, num_iters):
    """Fit HMM to FishPCDataset using expectation maximization.

    Parameters
        em_step: Callable[[GaussianHMM, np.ndarray] -> [GaussianHMM, NGSS]]
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        initial_hmm: GaussianHMM
        num_iters: int
            Number of EM steps to run.
    """

    for itr in range(num_iters):
        # Run full-batch E-step
        pass

def print_mem_reqs(ds: FishPCDataset, num_hmm_states: int, num_batches: int=1):
    # TODO num_batches
    tot_num_frames = np.sum(ds.num_frames)
    data_size = tot_num_frames * ds.dim * 4 / num_batches
    comp_size = tot_num_frames * num_hmm_states * 4 / num_batches

    to_gb = lambda sz : sz / (1024**3)
    print(f'Loading dataset with {tot_num_frames} frames, fitting HMM with {num_hmm_states:d} states, in {num_batches:d} batches.')
    print(f"Expecting {to_gb(data_size):.1f}GB req'd for loading data, {to_gb(comp_size):.1f}GB req'd for compute.")

def main():
    seed = jr.PRNGKey(45212276)
    batch_size = 6
    num_em_iters = 10
    num_hmm_states = 20
    method = 'nomap'
    
    seed, seed_split_data, seed_init_hmm = jr.split(seed, 3)

    # Initialize training and testing datasets
    print('Initializing training and testing datasets...')
    full_ds = FishPCDataset('fish0_137', DATADIR)
    train_ds, test_ds = full_ds.train_test_split(num_train=10,
                                                 num_test=1,
                                                 seed=seed_split_data)

    print('Train dataset mem requirements')
    print_mem_reqs(train_ds, num_hmm_states)
    
    print('Test dataset mem requirements')
    print_mem_reqs(test_ds, num_hmm_states)
    # Initialize hidden Markov model
    print(f'Initializing HMM with {num_hmm_states} states...')
    emission_dim = full_ds.dim
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm,
                                                 num_hmm_states,
                                                 emission_dim)

    # Fit
    if method == 'map':
        raise NotImplementedError
    elif method =='pmap':
        raise NotImplementedError
    elif method == 'nomap':
        train_lls, test_lls = fit_hmm_nomap(train_ds, test_ds, init_hmm, num_em_iters)
        print('train log likelihoods')
        print(train_lls)
        print('\n---------\n')
        print('test log likelihoods')
        print(test_lls)
    return

if __name__ == '__main__':
    main()