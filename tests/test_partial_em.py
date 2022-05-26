"""Ensure EM via partial E-step is equivalent to regular EM."""
import jax.config
jax.config.update('jax_platform_name', 'cpu')

from absl.testing import absltest
import chex

from kf.inference import partial_e_step, m_step

from functools import partial
from jax import vmap
import jax.numpy as np
import jax.random as jr

from ssm_jax.hmm.models import GaussianHMM
from ssm_jax.hmm.inference import hmm_smoother

# Suppress JAX/TFD warning: ...`check_dtpes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -----------------------------------------------------------------------------

def standard_e_step(hmm, emissions):
    return hmm_smoother(hmm.initial_probabilities,
                        hmm.transition_matrix,
                        hmm.emission_distribution.log_prob(emissions[..., None, :]))

def standard_m_step(posterior, emissions):
    from tensorflow_probability.substrates.jax.distributions import Dirichlet

    # Initial distribution
    initial_probs = Dirichlet(1.0001 + posterior.smoothed_probs[0]).mode()

    # Transition distribution
    transition_matrix = Dirichlet(
        1.0001 + np.einsum('tij->ij', posterior.smoothed_transition_probs)).mode()

    # Gaussian emission distribution
    w_sum = np.einsum('tk->k', posterior.smoothed_probs)
    x_sum = np.einsum('tk, ti->ki', posterior.smoothed_probs, emissions)
    xxT_sum = np.einsum('tk, ti, tj->kij', posterior.smoothed_probs, emissions, emissions)

    emission_means = x_sum / w_sum[:, None]
    emission_covs = xxT_sum / w_sum[:, None, None] \
        - np.einsum('ki,kj->kij', emission_means, emission_means) \
        + 1e-4 * np.eye(emissions.shape[1])
    
    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_probs,
                       transition_matrix,
                       emission_means,
                       emission_covs)

def standard_em_step(hmm, emissions):
    posterior = standard_e_step(hmm, emissions)
    hmm = standard_m_step(posterior, emissions)
    return hmm, posterior

class TestPartialEM(chex.TestCase):
    def setUp(self):
        super().setUp()

        seed_1, seed_2 = jr.split(jr.PRNGKey(60322))

        emission_dim = 2

        # -------------------------------
        # Make a true HMM
        # -------------------------------
        true_num_states = 6

        # Specify parameters
        initial_state_probs = np.ones(true_num_states) / true_num_states
        transition_matrix = 0.95 * np.eye(true_num_states) \
                            + 0.05 * np.roll(np.eye(true_num_states), 1, axis=1)
        
        emission_means = np.column_stack([
            np.cos(np.linspace(0, 2 * np.pi, true_num_states+1))[:-1],
            np.sin(np.linspace(0, 2 * np.pi, true_num_states+1))[:-1]
        ])
        emission_covs = np.tile(0.1**2 * np.eye(emission_dim),
                                (true_num_states, 1, 1))

        # Make a true HMM and sample
        true_hmm = GaussianHMM(initial_state_probs,
                               transition_matrix,
                               emission_means, 
                               emission_covs)
        true_states, emissions = true_hmm.sample(seed_1, 5000)

        self.true_states = true_states
        self.true_num_states = true_num_states
        self.emissions = emissions

        # -------------------------------
        num_states = 10
        init_hmm = GaussianHMM.random_initialization(seed_2, num_states, emission_dim)
        self.init_hmm = init_hmm
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepEvenSplit(self):
        """All batches have the same number of obs, i.e. (M, T_M, D)
        for M batches and T_M = num_obs // M"""
        
        # Standard EM
        ref_hmm, _ = standard_em_step(self.init_hmm, self.emissions)

        # -------------------------------
        # This EM code
        e_fn = self.variant(partial_e_step)
        m_fn = self.variant(m_step)

        # -------------
        num_batches = 5
        split_emissions = np.array(np.split(self.emissions, num_batches, axis=0))
        
        # ERROR: GaussianHMM object not recognized as pytree, even though it's registered.
        # weighted_suff_stats = vmap(partial_e_step, (None, 0))(init_hmm, split_emissions)
        weighted_suff_stats = vmap(partial(e_fn, self.init_hmm))(split_emissions)
        
        test_hmm = m_fn(*weighted_suff_stats)

        # ----------------------------------------------------------------------
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
        
        # Standard EM
        ref_hmm, _ = standard_em_step(self.init_hmm, self.emissions)
        
        # -------------------------------
        # This EM code
        e_fn = self.variant(partial_e_step)
        m_fn = self.variant(m_step)

        # -------------
        num_batches = 6
        i_tmp = len(self.emissions) % num_batches
        split_emissions = np.array_split(self.emissions, num_batches, axis=0)

        # 2 elements in list, each element has shape (m, K, D)
        split_suff_stats = [
                vmap(partial(e_fn, self.init_hmm))(np.array(e))       # vmap(partial(...)) is workaround
                for e in [split_emissions[:i_tmp],split_emissions[i_tmp:]]
        ]

        # ss elements have shape (M, K, ...)
        weighted_suff_stats = tuple([
                np.concatenate([ss0, ss1], axis=0)
                for ss0, ss1 in zip(*split_suff_stats)
        ])
        
        test_hmm = m_fn(*weighted_suff_stats)

        # ----------------------------------------------------------------------
        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-2))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-2))

    @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepUnevenSplit(self):
        """Batches have non-identical and not similiar number of obs, i.e. 
        (M, T_m, D) for M batches and T_m is the number of obs for batch m.
        
        We use the np.array_split(arr, n) function, which returns a list of
        arrays such that
            - len(arr) % n subarrays have size len(arr)//n + 1, and
            - the rest of the subarrays have size l//n
        """
        
        # Standard EM
        ref_hmm, _ = standard_em_step(self.init_hmm, self.emissions)
        
        # -------------------------------
        # This EM code
        e_fn = self.variant(partial_e_step)
        m_fn = self.variant(m_step)

        # -------------
        # Results in 5 arrays with lengths (300, 930, 2748, 972, 50)
        i_splits = np.array([300, 1230, 3978, 4950])

        # M elements in list, each element has shape (K,D)
        split_suff_stats = [
                e_fn(self.init_hmm, e)
                for e in np.split(self.emissions, i_splits, axis=0)
        ]

        # ss elements have shape (M, K, ...)
        weighted_suff_stats = tuple([
                np.stack(sss, axis=0)
                for sss in zip(*split_suff_stats)
        ])
        
        test_hmm = m_fn(*weighted_suff_stats)

        # ----------------------------------------------------------------------
        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-3))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-3))

if __name__ == '__main__':
    absltest.main()