"""Ensure EM via partial E-step is equivalent to regular EM."""
import jax.config
jax.config.update('jax_platform_name', 'cpu')

from absl.testing import absltest
import chex

from kf.inference import (sharded_e_step, collective_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS,
                          fullbatch_e_step, fullbatch_m_step)

from functools import partial, reduce
from jax import vmap
import jax.numpy as np
import jax.random as jr

from ssm_jax.hmm.models import GaussianHMM

# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -----------------------------------------------------------------------------

def fullbatch_em_step(hmm: GaussianHMM, emissions: chex.Array):
    posterior = fullbatch_e_step(hmm, emissions)
    hmm = fullbatch_m_step(posterior, emissions)
    return hmm, posterior

def get_leading_dim(nss: NGSS) -> chex.Array:
    return np.array([len(nss[k]) for k in nss.__dataclass_fields__.keys()])
    
class TestPartialEM(chex.TestCase):
    def setUp(self):
        super().setUp()

        seed_1, seed_2 = jr.split(jr.PRNGKey(60322))
        emission_dim = 2

        # -------------------------------
        # Make a true HMM and sample
        true_num_states = 6

        initial_state_probs = np.ones(true_num_states) / true_num_states
        transition_matrix = 0.95 * np.eye(true_num_states) \
                            + 0.05 * np.roll(np.eye(true_num_states), 1, axis=1)
        
        emission_means = np.column_stack([
            np.cos(np.linspace(0, 2 * np.pi, true_num_states+1))[:-1],
            np.sin(np.linspace(0, 2 * np.pi, true_num_states+1))[:-1]
        ])
        emission_covs = np.tile(0.1**2 * np.eye(emission_dim),
                                (true_num_states, 1, 1))

        true_hmm = GaussianHMM(initial_state_probs,
                               transition_matrix,
                               emission_means, 
                               emission_covs)
        true_states, emissions = true_hmm.sample(seed_1, 5000)

        self.emissions = emissions

        # -------------------------------
        # Randomly initialize a test HMM
        num_states = 10
        self.init_hmm = \
            GaussianHMM.random_initialization(seed_2, num_states, emission_dim)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepEvenSplit(self):
        """All batches have the same number of obs, i.e. (M, T_M, D)
        for M batches and T_M = num_obs // M"""
        
        def _em_step(hmm, split_emissions):
            # Since we vmap'd arrays directly, results in a single SuffStats
            # instance with batch shape (M,...). No addt'l transforms req'd.
            normd_suff_stats = vmap(partial(sharded_e_step, hmm))(split_emissions)
            
            new_hmm = collective_m_step(normd_suff_stats)

            return new_hmm, normd_suff_stats
        em_step = self.variant(_em_step)

        num_batches = 5
        split_emissions = np.array(np.split(self.emissions, num_batches, axis=0))
        test_hmm, test_nss = em_step(self.init_hmm, split_emissions)

        # ----------------------------------------------------------------------

        self.assertTrue(np.all(get_leading_dim(test_nss)==num_batches))

        ref_hmm, _ = fullbatch_em_step(self.init_hmm, self.emissions)
        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-3))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-3))

    @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepNearEvenSplit(self):
        """Batches have non-identical (but similar) number of obs, i.e. (M, T_m, D)
        for M batches and T_m is the number of obs for batch m.
        
        We use the np.array_split(arr, n) function, which returns a list of
        arrays such that
            - len(arr) % n subarrays have size len(arr)//n + 1, and
            - the rest of the subarrays have size l//n
        """
        def _em_step(hmm, split_emissions):
            # Since we vmap'd, each SuffStats class has batch shape (m,...).
            # So, concatenate together, resulting in instance with batch shape (M,...)
            e_step = partial(sharded_e_step, hmm)
            _normd_suff_stats = \
                [vmap(e_step)(np.array(se)) for se in split_emissions]
            normd_suff_stats = reduce(NGSS.concat, _normd_suff_stats)
            
            new_hmm = collective_m_step(normd_suff_stats)

            return new_hmm, normd_suff_stats
        em_step = self.variant(_em_step)

        num_batches = 6
        i_tmp = len(self.emissions) % num_batches
        split_emissions = np.array_split(self.emissions, num_batches, axis=0)

        test_hmm, test_nss = em_step(self.init_hmm,
                                      [split_emissions[:i_tmp], split_emissions[i_tmp:]])

        # ---------------------------------------------------------------------

        self.assertTrue(np.all(get_leading_dim(test_nss)==num_batches))

        # Compare results to standard EM
        ref_hmm, _ = fullbatch_em_step(self.init_hmm, self.emissions)
        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-2))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-2))

    @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepUnevenSplit(self):
        """Batches have non-identical and not similiar number of obs
        i.e. (M, T_m, D) for M batches and T_m is the number of obs for batch m.
        
        We use the np.array_split(arr, n) function, which returns a list of
        arrays such that
            - len(arr) % n subarrays have size len(arr)//n + 1, and
            - the rest of the subarrays have size l//n
        """
        
        def _em_step(hmm, split_emissions):
            # Each SuffStats class has leading shape (K,...) and not (batch, K,...)
            # since we didn't vmap or pmap it. So, stack them together.
            _normd_suff_stats = [sharded_e_step(hmm, se) for se in split_emissions]
            normd_suff_stats = NGSS.stack(_normd_suff_stats)
            
            new_hmm = collective_m_step(normd_suff_stats)

            return new_hmm, normd_suff_stats
        em_step = self.variant(_em_step)

        # Split emissions into 5 (len(i_splits)+1) arrays,
        # of lengths (300, 930, 2748, 972, 50)
        i_splits = np.array([300, 1230, 3978, 4950])
        split_emissions = np.split(self.emissions, i_splits, axis=0)
        num_batches = len(split_emissions)

        test_hmm, test_nss = em_step(self.init_hmm, split_emissions)

        # ----------------------------------------------------------------------
        self.assertTrue(np.all(get_leading_dim(test_nss)==num_batches))
        
        # Compare results to standard EM
        ref_hmm, _ = fullbatch_em_step(self.init_hmm, self.emissions)

        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-3))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-3))

if __name__ == '__main__':
    absltest.main()