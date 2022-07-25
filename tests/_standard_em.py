"""Reference Gaussian HMM EM code for evaluating test_inference.py results"""
import chex
from jax import vmap, tree_map
import jax.numpy as jnp
from functools import partial

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from ssm_jax.hmm.inference import hmm_smoother, compute_transition_probs

from tensorflow_probability.substrates.jax.distributions import Dirichlet

def standard_e_step(hmm: GaussianHMM, batch_emissions: chex.Array):
    @chex.dataclass
    class GaussianHMMSuffStats:
        # Wrapper for sufficient statistics of a GaussianHMM
        marginal_loglik: chex.Scalar
        initial_probs: chex.Array
        trans_probs: chex.Array
        sum_w: chex.Array
        sum_x: chex.Array
        sum_xxT: chex.Array

    def _single_e_step(emissions):
        # Run the smoother
        posterior = hmm_smoother(
            hmm.initial_probabilities,
            hmm.transition_matrix,
            hmm._conditional_logliks(emissions)
        )

        # Compute the initial state and transition probabilities
        initial_probs = posterior.smoothed_probs[0]
        trans_probs = compute_transition_probs(hmm.transition_matrix, posterior)

        # Compute the expected sufficient statistics
        sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
        sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)
        sum_xxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, emissions, emissions)

        stats = GaussianHMMSuffStats(
            marginal_loglik=posterior.marginal_loglik,
            initial_probs=initial_probs,
            trans_probs=trans_probs,
            sum_w=sum_w,
            sum_x=sum_x,
            sum_xxT=sum_xxT
        )
        return stats

    # Map the E step calculations over batches
    return vmap(_single_e_step)(batch_emissions)

def standard_m_step(batch_emissions, batch_posteriors):
     # Sum the statistics across all batches
    stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

    # Initial distribution
    initial_probs = Dirichlet(1.0001 + stats.initial_probs).mode()

    # Transition distribution
    transition_matrix = Dirichlet(1.0001 + stats.trans_probs).mode()

    # Gaussian emission distribution
    emission_dim = stats.sum_x.shape[-1]
    emission_means = stats.sum_x / stats.sum_w[:, None]
    emission_covs = (
        stats.sum_xxT / stats.sum_w[:, None, None]
        - jnp.einsum("ki,kj->kij", emission_means, emission_means)
        + 1e-4 * jnp.eye(emission_dim)
    )
    
    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_probs,
                       transition_matrix,
                       emission_means,
                       emission_covs)

def standard_em_step(hmm, emissions):
    posterior = standard_e_step(hmm, emissions)
    hmm = standard_m_step(emissions, posterior)
    return hmm, posterior