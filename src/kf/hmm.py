"""Customized GaussianHMM class

Notable changes from the parent GaussianHMM class (from ssm-jax) include:
- Computes normalized sufficient statistics in E-step, instead of in M-step
- Implements stochastic EM with weighted sufficient statistics
- TODO Save and load from .npz file, using Parameters
"""

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from jax import vmap, lax
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map

from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.distributions import NormalInverseWishart, niw_posterior_update
from ssm_jax.hmm.models import GaussianHMM as StandardGaussianHMM

__all__ = [
    'NormalizedGaussianHMMSuffStats',
    'GaussianHMM',
]

@chex.dataclass
class NormalizedGaussianHMMSuffStats:
    marginal_loglik: chex.Array # scalar array
    initial_probs: chex.Array
    trans_probs: chex.Array
    weights: chex.Array         # still equivalent to sum_w
    normd_x: chex.Array
    normd_xxT: chex.Array

    def reduce(self, axis=0):
        """Reduce batched instance along specified axis."""
        
        total_weights = self.weights.sum(axis=axis)
        normd_weights = self.weights / total_weights
        normd_x = (normd_weights[...,None] * self.normd_x).sum(axis=axis)
        normd_xxT = (normd_weights[...,None,None] * self.normd_xxT).sum(axis=axis)

        return NormalizedGaussianHMMSuffStats(
            marginal_loglik=self.marginal_loglik.sum(axis=axis),
            initial_probs=self.initial_probs.sum(axis=axis),
            trans_probs=self.trans_probs.sum(axis=axis),
            weights=total_weights,
            normd_x=normd_x,
            normd_xxT=normd_xxT,
        )

    def __add__(self, other):
        total_weights = self.weights + other.weights
        
        these_weights = self.weights/total_weights
        other_weights = other.weights/total_weights
        
        normd_x = these_weights[...,None] * self.normd_x \
                  + other_weights[...,None] * other.normd_x
        normd_xxT = these_weights[...,None,None] * self.normd_xxT \
                    + other_weights[...,None,None] * other.normd_xxT

        return NormalizedGaussianHMMSuffStats(
            marginal_loglik=self.marginal_loglik+other.marginal_loglik,
            initial_probs=self.initial_probs+other.initial_probs,
            trans_probs=self.trans_probs+other.trans_probs,
            weights=total_weights,
            normd_x=normd_x,
            normd_xxT=normd_xxT,
        )
    
    def __radd__(self, other):
        return lax.cond(
            other==0,
            lambda x: x,
            lambda x: x.__add__(other),
            self
        )
        
@register_pytree_node_class
class GaussianHMM(StandardGaussianHMM):
    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrices,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):

        super().__init__(
            initial_probabilities,
            transition_matrix,
            emission_means,
            emission_covariance_matrices,
            initial_probs_concentration,
            transition_matrix_concentration,
            emission_prior_mean,
            emission_prior_concentration,
            emission_prior_scale,
            emission_prior_extra_df,
        )
    
    def _zeros_like_suff_stats(self):
        dim = self.num_obs
        num_states = self.num_states
        return NormalizedGaussianHMMSuffStats(
            marginal_loglik=jnp.zeros(()),
            initial_probs=jnp.zeros((num_states,)),
            trans_probs=jnp.zeros((num_states, num_states)),
            weights=jnp.zeros((num_states,)),
            normd_x=jnp.zeros((num_states, dim)),
            normd_xxT=jnp.zeros((num_states, dim, dim)),
        )
    
    def e_step(self, batch_emissions):
        """Compute the expected sufficient statistics under the posterior.
        
        The expected sufficient statistics of the emissions (which are weighted
        by the posterior smoothed likelihoods) are normalized by the total
        likelihood over time (summed_weights). This is more numerically stable.

        Args
            batch_emissions, ndarray, shape (batch_size, num_timesteps, obs_dim)
        Returns
            NormalizedGaussianHMMSuffStats, dataclass with leading (batch_size,)
        """

        def _single_e_step(emissions):
            # Run the smoother to calculate the posterior
            posterior = hmm_smoother(self._compute_initial_probs(), self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute normalized weights, catch DBZ errors
            summed_weights = jnp.einsum("tk->k", posterior.smoothed_probs)
            normd_weights = jnp.where(
                summed_weights[None,:] > 0.,
                posterior.smoothed_probs / summed_weights,
                0.
            )

            # Compute the normalized expected sufficient statistics
            normd_x = jnp.einsum("tk,ti->ki", normd_weights, emissions)
            normd_xxT = jnp.einsum("tk,ti,tj->kij", normd_weights, emissions, emissions)

            return NormalizedGaussianHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=posterior.initial_probs,
                trans_probs=trans_probs,
                weights=summed_weights,
                normd_x=normd_x,
                normd_xxT=normd_xxT
            )

        # Map the E step calculations over the batch_size dimension
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_posteriors, **kwargs):
        # Reduce the statistics along the batch dimension
        stats = batch_posteriors.reduce()

        # TODO NIW posterior update can be simplified since we are using normd SS
        # The expected log joint is equal to the log prob of a normal inverse
        # Wishart distribution, up to additive factors. Find this NIW distribution
        # take its mode.
        niw_prior = NormalInverseWishart(loc=self._emission_prior_mean.value,
                                         mean_concentration=self._emission_prior_conc.value,
                                         df=self._emission_prior_df.value,
                                         scale=self._emission_prior_scale.value)

        # loc_pri, precision_pri, df_pri, scale_pri = niw_prior.parameters.values()
        Sx, SxxT, N = stats.normd_x, stats.normd_xxT, stats.weights
        Sx *= N[:,None]
        SxxT *= N[:,None,None]

        # Find the posterior parameters of the NIW distribution
        def _single_m_step(sum_w, sum_x, sum_xxT):
            niw_posterior = niw_posterior_update(niw_prior, (sum_x, sum_xxT, sum_w))
            return niw_posterior.mode()

        covs, means = vmap(_single_m_step)(N, Sx, SxxT)
        self.emission_covariance_matrices.value = covs
        self.emission_means.value = means
    
    