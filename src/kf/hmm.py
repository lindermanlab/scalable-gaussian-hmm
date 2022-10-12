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
from functools import partial

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
    """All sufficienti stats are weighted by `weights`."""
    initial_probs: chex.Array   # (K,)
    trans_probs: chex.Array     # (K,K)
    normd_x: chex.Array         # (K,D)
    normd_xxT: chex.Array       # (K,D,D)
    weights: chex.Array         # (K,)

    def reduce(self, axis=0, keepdims=False):
        """Perform weighted summation along the indicated axes.
        Does NOT check if dataclass is already minimal (i.e. batch_shape=()).
        """
        
        total_weights = self.weights.sum(axis=axis, keepdims=keepdims)
        normd_weights = self.weights / total_weights
        _reduce = (lambda ss, shp: 
            (jnp.expand_dims(normd_weights, shp)*ss).sum(axis=axis, keepdims=keepdims)
        )

        return NormalizedGaussianHMMSuffStats(
            initial_probs=self.initial_probs.sum(axis=axis, keepdims=keepdims),
            trans_probs=self.trans_probs.sum(axis=axis, keepdims=keepdims),
            normd_x=_reduce(self.normd_x, (-1)),
            normd_xxT=_reduce(self.normd_xxT, (-1,-2)),
            weights=total_weights,
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
        K, D = self.num_states, self.num_obs
        return NormalizedGaussianHMMSuffStats(
            initial_probs=jnp.zeros((K,)),
            trans_probs=jnp.zeros((K, K)),
            normd_x=jnp.zeros((K, D)),
            normd_xxT=jnp.zeros((K, D, D)),
            weights=jnp.zeros((K,)),
        )
    
    def e_step(self, batch_emissions):
        """Compute the expected sufficient statistics under the posterior.

        The expected sufficient statistics of the emissions (which are weighted
        by the posterior smoothed likelihoods) are normalized by the number of 
        of emissions used to calculate the likelihood. This is more numerically stable.

        Args
            batch_emissions, ndarray, shape (batch_size, num_timesteps, obs_dim)

        Returns
            NormalizedGaussianHMMSuffStats, dataclass with leading (batch_size,...)
            posterior_marginal_logliklihood, array with shape (batch_size,)
        """

        def _single_e_step(emissions):
            # Run the smoother to calculate the posterior
            posterior = hmm_smoother(self._compute_initial_probs(), self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute normalized weights
            total_weights = jnp.einsum("tk->k", posterior.smoothed_probs)       # shape (K,)
            normd_weights = jnp.where(                                          # shape (T,K)
                total_weights[None,:] > 0.,  
                posterior.smoothed_probs / total_weights, 
                0.)
            
            # Compute normalized expected sufficient statistics, store normalizer
            normd_stats = NormalizedGaussianHMMSuffStats(
                initial_probs=posterior.initial_probs,
                trans_probs=trans_probs,
                normd_x=jnp.einsum("tk,ti->ki", normd_weights, emissions),
                normd_xxT=jnp.einsum("tk,ti,tj->kij", normd_weights, emissions, emissions),
                weights=total_weights,
            )

            return (normd_stats, posterior.marginal_loglik)

        # Map the E step calculations over the batch_size dimension
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_emission_stats, **kwargs):
        """Calculate emission parameter mode directly from normalized posterior."""

        # Reduce stats along the batch dimension. Fields have leading shape (K,...)
        normd_emission_stats = batch_emission_stats.reduce()
        
        # Normalize the natural params of the prior
        natural_prior = niw_mean_to_natural_params(
            self._emission_prior_mean.value,
            self._emission_prior_conc.value,
            self._emission_prior_scale.value,
            self._emission_prior_df.value
            )

        normd_natural_prior = tree_map(
            lambda param, shp: param/jnp.expand_dims(normd_emission_stats.weights, shp),
            natural_prior, ((), (-1,-2), (-1), ())
        )

        # Calculate the natural parameters of the posterior
        normd_natural_posterior = tree_map(
            jnp.add,
            normd_natural_prior,
            (jnp.ones(self.num_states), normd_emission_stats.normd_xxT, normd_emission_stats.normd_x, jnp.ones(self.num_states))
        )

        # Compute modal mean and covariance parameters for each state
        def _single_m_step(normd_eta_1, normd_eta_2, normd_eta_3, normd_eta_4):
            loc, _, scale, _ = niw_natural_to_mean_params(
                    normd_eta_1, normd_eta_2, normd_eta_3, normd_eta_4
            )
            cov = scale / normd_eta_1
            return loc, cov
        
        # Map the M-step over the states dimension
        means, covs = vmap(_single_m_step)(*normd_natural_posterior)
        self.emission_covariance_matrices.value = covs
        self.emission_means.value = means

    def fit_em(self, batch_emissions, num_iters=50, **kwargs):
        """Fit this HMM with Expectation-Maximization (EM)."""
        @jit
        def em_step(params):
            self.unconstrained_params = params
            batch_posteriors, batch_marginal_lls = self.e_step(batch_emissions)
            lp = self.log_prior() + batch_marginal_lls.sum()
            self.m_step(batch_emissions, batch_posteriors, **kwargs)
            return self.unconstrained_params, lp

        log_probs = []
        params = self.unconstrained_params
        for _ in trange(num_iters):
            params, lp = em_step(params)
            log_probs.append(lp)

        self.unconstrained_params = params
        return jnp.array(log_probs)

    def fit_stochastic_em(self, emissions_generator, total_emissions, schedule=None, nepochs=50):
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
            nepochs (int): Num of iterations made through the entire dataset.
        Returns:
            expected_log_prob (chex.Array): Mean expected log prob of each epoch.
        """

        nbatches = len(emissions_generator)

        # Set global training learning rates: shape (nepochs, nbatches)
        if schedule is None:
            schedule = optax.exponential_decay(
                init_value=1.,
                end_value=0.,
                transition_steps=nbatches,
                decay_rate=.95,
            )

        learning_rates = schedule(jnp.arange(nepochs * nbatches))
        assert learning_rates[0] == 1.0, "Learning rate must start at 1."
        learning_rates = learning_rates.reshape(nepochs, nbatches)

        # @jit
        def minibatch_em_step(carry, inputs):
            params, normd_rolling_stats = carry
            minibatch_emissions, learning_rate = inputs

            # Compute the sufficient stats given a minibatch of emissions
            self.unconstrained_params = params
            normd_minibatch_stats, minibatch_marginal_lls = self.e_step(minibatch_emissions)

            # Reduce batched stats so that fields have leading shape (K,...)
            normd_minibatch_stats = normd_minibatch_stats.reduce()
            
            # Calculate expected log probability of the scaled minibatch
            expected_lp = self.log_prior() + nbatches * minibatch_marginal_lls.sum()

            # Incorporate minibatch stats into the rolling averaged stats
            def _roll_in(rr, bb): 
                # update function for unnormalized / summed statistics
                _update = (lambda rolling, batch:
                    (1-learning_rate) * rolling + learning_rate * nbatches * batch
                )
                
                # update function for normalized statistics
                w_r = (1-learning_rate) * rr.weights
                w_b = learning_rate * nbatches * bb.weights
                w_s = w_r + w_b

                _weighted_update = (lambda rolling, batch:
                    jnp.einsum('k,k...->k...', w_r/w_s, rolling)
                    + jnp.einsum('k,k...->k...', w_b/w_s, batch)
                )

                return NormalizedGaussianHMMSuffStats(
                    initial_probs=_update(rr.initial_probs, bb.initial_probs),
                    trans_probs=_update(rr.trans_probs, bb.trans_probs),
                    normd_x=_weighted_update(rr.normd_x, bb.normd_x),
                    normd_xxT=_weighted_update(rr.normd_xxT, bb.normd_xxT),
                    weights=w_s,
                )

            normd_rolling_stats = _roll_in(normd_rolling_stats, normd_minibatch_stats)            
            
            # Call M-step
            batch_normd_rolling_stats = tree_map(
                partial(jnp.expand_dims, axis=0), normd_rolling_stats
            )
            self.m_step(minibatch_emissions, batch_normd_rolling_stats)

            return (self.unconstrained_params, normd_rolling_stats), expected_lp

        # Initialize and train
        expected_log_probs = jnp.empty((0, len(emissions_generator)))
        params = self.unconstrained_params
        normd_rolling_stats = self._zeros_like_suff_stats()
        for epoch in trange(nepochs):
            epoch_expected_lps = []
            for minibatch, minibatch_emissions in enumerate(emissions_generator):
                (params, normd_rolling_stats), minibatch_expected_lp = \
                    minibatch_em_step(
                        (params, normd_rolling_stats),
                        (minibatch_emissions, learning_rates[epoch][minibatch]),
                    )
                epoch_expected_lps.append(minibatch_expected_lp)

            # Save epoch mean of expected log probs
            expected_log_probs = jnp.vstack([expected_log_probs, jnp.asarray(epoch_expected_lps)])

        # Update self with fitted params
        self.unconstrained_params = params
        return jnp.asarray(expected_log_probs)