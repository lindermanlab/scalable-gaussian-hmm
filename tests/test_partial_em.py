"""Ensure EM via partial E-step is equivalent to regular EM."""
import jax.config
jax.config.update('jax_platform_name', 'cpu')

from absl.testing import absltest
import chex

from kf.inference import partial_e_step, m_step

from jax import vmap
from functools import partial

import jax.numpy as np
import jax.random as jr
from ssm_jax.hmm.models import GaussianHMM
from ssm_jax.hmm.inference import hmm_smoother
from tensorflow_probability.substrates.jax.distributions import Dirichlet

# Suppress JAX/TFD warning: ...`check_tpes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())


def standard_e_step(hmm, emissions):
    return hmm_smoother(hmm.initial_probabilities,
                        hmm.transition_matrix,
                        hmm.emission_distribution.log_prob(emissions[..., None, :]))

def standard_m_step(posterior, emissions):
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

        num_states = 6
        emission_dim = 2

        # Specify parameters of the HMM
        initial_state_probs = np.ones(num_states) / num_states
        transition_matrix = 0.95 * np.eye(num_states) \
                            + 0.05 * np.roll(np.eye(num_states), 1, axis=1)
        
        emission_means = np.column_stack([
            np.cos(np.linspace(0, 2 * np.pi, num_states+1))[:-1],
            np.sin(np.linspace(0, 2 * np.pi, num_states+1))[:-1]
        ])
        emission_covs = np.tile(0.1**2 * np.eye(emission_dim), (num_states, 1, 1))

        # Make a true HMM and sample
        true_hmm = GaussianHMM(initial_state_probs,
                               transition_matrix,
                               emission_means, 
                               emission_covs)
        true_states, emissions = true_hmm.sample(jr.PRNGKey(60322), 5000)

        # self.true_hmm = true_hmm
        self.true_states = true_states
        self.true_num_states = num_states
        self.emissions = emissions
        
    # @chex.variants(with_jit=True, without_jit=True)
    def testSingleStepEvenSplit(self):
        num_states = 10
        emission_dim = self.emissions.shape[-1]
        init_hmm = GaussianHMM.random_initialization(jr.PRNGKey(381064),
                                                     num_states, emission_dim)

        # Standard EM
        ref_hmm, _ = standard_em_step(init_hmm, self.emissions)
        
        # This EM code
        num_batches = 5
        split_emissions = np.array(np.split(self.emissions, num_batches, axis=0))
        weighted_suff_stats = vmap(partial(partial_e_step, init_hmm))(split_emissions)
        test_hmm = m_step(*weighted_suff_stats)

        # ----------------------------------------------------------------------
        self.assertTrue(np.allclose(ref_hmm.emission_means,
                                    test_hmm.emission_means,
                                    atol=1e-3))
        
        self.assertTrue(np.allclose(ref_hmm.emission_covariance_matrices,
                                    test_hmm.emission_covariance_matrices,
                                    atol=1e-3))

if __name__ == '__main__':
    absltest.main()