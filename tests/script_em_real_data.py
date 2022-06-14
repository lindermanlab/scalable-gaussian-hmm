# Time EM on real data

import os

from jax import vmap, pmap, lax
import jax.numpy as np
import jax.random as jr
from functools import partial, reduce

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step as e_step,
                          collective_m_step as m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)

from kf.data_utils import FishPCDataset, FishPCDataloader

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

    for itr in trange(num_iters):
        # Run full-batch E-step
        pass      

def main():
    seed = jr.PRNGKey(45212276)
    batch_size = 6
    num_hmm_states = 20
    
    seed, seed_split_data, seed_init_hmm = jr.split(seed, 3)

    # Initialize training and testing datasets
    full_ds = FishPCDataset('fish0_137', DATADIR)
    train_ds, test_ds = full_ds.train_test_split(num_train=100,
                                                 num_test=10,
                                                 seed=seed_split_data)
    # Initialize hidden Markov model
    emission_dim = full_ds.dim
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm,
                                                 num_hmm_states,
                                                 emission_dim)

    # Fit

    return

if __name__ == '__main__':
    main()