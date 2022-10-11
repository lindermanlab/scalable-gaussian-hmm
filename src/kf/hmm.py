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
import optax

from jax import vmap, lax, jit
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map

from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.distributions import NormalInverseWishart, niw_posterior_update
from ssm_jax.hmm.models import GaussianHMM as StandardGaussianHMM

from tqdm.auto import trange

__all__ = [
    'NormalizedGaussianHMMSuffStats',
    'GaussianHMM',
]

def niw_mean_to_natural_params(loc, conc, scale, df):
    dim = loc.shape[-1]
    eta_1 = df + dim + 2
    eta_2 = scale + conc * jnp.outer(loc, loc)
    eta_3 = conc * loc
    eta_4 = conc
    return (eta_1, eta_2, eta_3, eta_4)
        
def niw_natural_to_mean_params(eta_1, eta_2, eta_3, eta_4):
    dim = eta_3.shape[-1]
    loc = eta_3 / eta_4
    conc = eta_4            
    scale = eta_2 - jnp.outer(eta_3, eta_3) / eta_4
    df = eta_1 - dim - 2
    return (loc, conc, scale, df)

@chex.dataclass
class NormalizedGaussianHMMSuffStats:
    marginal_loglik: chex.Array # scalar array
    initial_probs: chex.Array
    trans_probs: chex.Array
    weights: chex.Array         # still equivalent to sum_w
    normd_x: chex.Array
    normd_xxT: chex.Array

    def reduce(self, axis=0, keepdims=False):
        """Perform weighted summation along specified axis to reduce batch instance."""
        
        total_weights = self.weights.sum(axis=axis, keepdims=keepdims)
        normd_weights = self.weights / total_weights
        normd_x = (normd_weights[...,None] * self.normd_x).sum(axis=axis, keepdims=keepdims)
        normd_xxT = (normd_weights[...,None,None] * self.normd_xxT).sum(axis=axis, keepdims=keepdims)

        return NormalizedGaussianHMMSuffStats(
            marginal_loglik=self.marginal_loglik.sum(axis=axis, keepdims=keepdims),
            initial_probs=self.initial_probs.sum(axis=axis, keepdims=keepdims),
            trans_probs=self.trans_probs.sum(axis=axis, keepdims=keepdims),
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
        by the posterior smoothed likelihoods) are normalized by the number of 
        of emissions used to calculate the likelihood. This is more numerically stable.

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

            # ---------------------------
            # COMPUTE NORMALIZED WEIGHTS
            # ---------------------------
            total_weights = jnp.einsum("tk->k", posterior.smoothed_probs)       # shape (K,)
            normd_weights = jnp.where(                                          # shape (T,K)
                total_weights[None,:] > 0.,  
                posterior.smoothed_probs / total_weights, 
                0.)
            
            # vvv THIS BREAKS THE INFERENCE vvv .. why??
            # Equivalent to each emissions having a weight of 1
            # total_weights = jnp.ones(self.num_states) * len(emissions)
            # total_weights = jnp.ones(self.num_states) * 200
            # normd_weights = posterior.smoothed_probs / total_weights
    
            # Compute the normalized expected sufficient statistics
            normd_x = jnp.einsum("tk,ti->ki", normd_weights, emissions)
            normd_xxT = jnp.einsum("tk,ti,tj->kij", normd_weights, emissions, emissions)

            return NormalizedGaussianHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=posterior.initial_probs,
                trans_probs=trans_probs,
                weights=total_weights,
                normd_x=normd_x,
                normd_xxT=normd_xxT,
            )

        # Map the E step calculations over the batch_size dimension
        return vmap(_single_e_step)(batch_emissions)

    # def _m_step_emissions(self, batch_emissions, batch_emission_stats, **kwargs):
    #     """NAIVE METHOD: Unnormalize sufficient stats"""
        # emission_stats = batch_emission_stats.reduce()

        # # The expected log joint is equal to the log prob of a NIW distribution,
        # # up to additive factors. Find this NIW distribution and take its mode.
        # niw_prior = NormalInverseWishart(loc=self._emission_prior_mean.value,
        #                                  mean_concentration=self._emission_prior_conc.value,
        #                                  df=self._emission_prior_df.value,
        #                                  scale=self._emission_prior_scale.value)

        # # Find the posterior parameters of the NIW distribution
        # def _single_m_step(sum_x, sum_xxT, sum_w):
        #     niw_posterior = niw_posterior_update(niw_prior, (sum_x, sum_xxT, sum_w))
        #     return niw_posterior.mode()

        # import pdb; pdb.set_trace()

        # # NAIVE/ORIGINAL APPROACH: Unnormalize sufficient statistics
        # Sx, SxxT, N = emission_stats.normd_x, emission_stats.normd_xxT, emission_stats.weights
        # Sx *= N[:,None]
        # SxxT *= N[:,None,None]

        # covs, means = vmap(_single_m_step)(Sx, SxxT, N)
        # self.emission_covariance_matrices.value = covs
        # self.emission_means.value = means

    def _m_step_emissions(self, batch_emissions, batch_emission_stats, **kwargs):
        """WISE METHOD: Calculate mode directly from normalized natural stats"""

        emission_stats = batch_emission_stats.reduce()
        normd_suff_stats = (jnp.ones(self.num_states),
                            emission_stats.normd_xxT,
                            emission_stats.normd_x,
                            jnp.ones(self.num_states)
                            )

        # Normalize the natural prior params by cumulative state weight. N has shape (K,)
        N = emission_stats.weights
        natural_prior = niw_mean_to_natural_params(
            self._emission_prior_mean.value,
            self._emission_prior_conc.value,
            self._emission_prior_scale.value,
            self._emission_prior_df.value
            )
            
        normd_natural_prior = tree_map(
            lambda param, n: param/n,
            natural_prior, (N, N[:,None,None], N[:,None], N)
        )

        # Calculate the natural posterior parameters
        normd_natural_posterior = tree_map(jnp.add, normd_suff_stats, normd_natural_prior)

        # Compute modal mean and covariance parameters for each state
        def _single_m_step(normd_eta_1, normd_eta_2, normd_eta_3, normd_eta_4):
            loc, _, scale, _ = niw_natural_to_mean_params(
                normd_eta_1, normd_eta_2, normd_eta_3, normd_eta_4
            )
            cov = scale / normd_eta_1
            return loc, cov
        
        means, covs = vmap(_single_m_step)(*normd_natural_posterior)

        self.emission_covariance_matrices.value = covs
        self.emission_means.value = means

    def fit_em(self, batch_emissions, num_iters=50, **kwargs):
        """Fit this HMM with Expectation-Maximization (EM)."""
        @jit
        def em_step(params):
            self.unconstrained_params = params
            batch_posteriors = self.e_step(batch_emissions)
            lp = self.log_prior() + batch_posteriors.marginal_loglik.sum()
            self.m_step(batch_emissions, batch_posteriors, **kwargs)
            return self.unconstrained_params, lp

        log_probs = []
        params = self.unconstrained_params
        for _ in trange(num_iters):
            params, lp = em_step(params)
            log_probs.append(lp)

        self.unconstrained_params = params
        return jnp.array(log_probs)

    def fit_stochastic_em(self, emissions_generator, total_emissions, schedule=None, num_epochs=50):
        """Fit this HMM by running Stochastic Expectation-Maximization.

        Assuming the original dataset consists of B*M independent sequences of
        length T, this algorithm performs EM on a random subset of M sequences
        (not timesteps) at each step. Importantly, the subsets of M sequences
        are shuffled at each epoch. It is up to the user to correctly
        instantiate the Dataloader generator object to exhibit this property.

        The algorithm uses a learning rate schedule to anneal the minibatch
        sufficient statistics at each stage of training. If a schedule is not
        specified, an exponentially decaying model is used such that the
        learning rate which decreases by 5% at each epoch.

        This algoirthm differs from that in the ExponentialFamilyHMM because it
        calls an E-step which returns normalized weighted sufficient statistics.

        Args:
            emissions_generator: Iterable over the emissions dataset, produces
                B minibatches of shape (M,T,D); auto-shuffles the loading order
                of mini-batches after each epoch.
            total_emissions (int): Total number of emissions that the generator
                will load. Used to scale the minibatch statistics.
            schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
                schedule; defaults to exponential schedule.
            num_epochs (int): Num of iterations made through the entire dataset.
        Returns:
            expected_log_prob (chex.Array): Mean expected log prob of each epoch.
        """

        num_batches = len(emissions_generator)

        # Set global training learning rates: shape (num_epochs, num_batches)
        if schedule is None:
            schedule = optax.exponential_decay(
                init_value=1.,
                end_value=0.,
                transition_steps=num_batches,
                decay_rate=.95,
            )

        learning_rates = schedule(jnp.arange(num_epochs * num_batches))
        assert learning_rates[0] == 1.0, "Learning rate must start at 1."
        learning_rates = learning_rates.reshape(num_epochs, num_batches)

        # @jit
        def minibatch_em_step(carry, inputs):
            params, normd_rolling_stats = carry
            minibatch_emissions, learning_rate = inputs

            # Compute the sufficient stats given a minibatch of emissions
            self.unconstrained_params = params
            normd_minibatch_stats = self.e_step(minibatch_emissions)            # leadig shape (B,K,...)
            normd_minibatch_stats = normd_minibatch_stats.reduce()              # leading shape (K,...)
            

            # Incorporate the minibatch stats into the rolling averaged stats
            normd_rolling_stats = tree_map(
                lambda ss0, ss1: (1-learning_rate) * ss0 + learning_rate * ss1,
                normd_rolling_stats, normd_minibatch_stats
            )

            # TODO should I divide log_prior by total number of emissions?
            # previously, this expected lp was calculated from scaled_minibatch_stats
            # (i.e scaled to be from the whole dataset). this normalized version is
            # more akin to...average margin loglik per emission?
            expected_lp = self.log_prior() + normd_minibatch_stats.marginal_loglik

            # Add a batch dimension and call M-step
            # TODO can this be simplified since we're performing our own M-step
            # batched_rolling_stats = tree_map(
            #             lambda x: jnp.expand_dims(x, axis=0), normd_rolling_stats
            # )
            batched_rolling_stats = normd_rolling_stats
            self.m_step(minibatch_emissions, batched_rolling_stats)

            return (self.unconstrained_params, normd_rolling_stats), expected_lp

        # Initialize and train
        expected_log_probs = []
        params = self.unconstrained_params
        normd_rolling_stats = self._zeros_like_suff_stats()
        for epoch in trange(num_epochs):

            epoch_expected_lp = 0.
            for minibatch, minibatch_emissions in enumerate(emissions_generator):
                (params, normd_rolling_stats), minibatch_expected_lp = \
                    minibatch_em_step(
                        (params, normd_rolling_stats),
                        (minibatch_emissions, learning_rates[epoch][minibatch]),
                    )
                epoch_expected_lp += minibatch_expected_lp

            # Save epoch mean of expected log probs
            expected_log_probs.append(epoch_expected_lp / num_batches)

        # Update self with fitted params
        self.unconstrained_params = params
        return jnp.array(expected_log_probs)