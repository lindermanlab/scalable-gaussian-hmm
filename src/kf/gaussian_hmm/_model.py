from typing import NamedTuple
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax

from tensorflow_probability.substrates.jax.distributions import (
    Dirichlet, Categorical, MultivariateNormalFullCovariance as MVNFull)
from dynamax.distributions import NormalInverseWishart

__all__ = [
    'Parameters',
    'PriorParameters',
    'HiddenMarkovChainStatistics',
    'NormalizedEmissionStatistics',
    'initial_distribution',
    'transition_distribution',
    'emission_distribution',
    'log_prior',
    'log_prob',
    'log_likelihood',
    'sample',
]

# Parameters and statistics for a Gaussian HMM with a Dirichlet prior
# on the hidden Markov Chain and normal inverse wishart prior on emissions

class Parameters(NamedTuple):
    initial_probs: jnp.ndarray
    transition_probs: jnp.ndarray
    emission_means: jnp.ndarray
    emission_covariances: jnp.ndarray

class PriorParameters(NamedTuple):
    initial_probs_conc: jnp.ndarray
    transition_probs_conc: jnp.ndarray
    emission_loc: jnp.ndarray
    emission_conc: jnp.ndarray
    emission_scale: jnp.ndarray
    emission_df: jnp.ndarray

class HiddenMarkovChainStatistics(NamedTuple):
    initial_pseudocounts: jnp.ndarray
    transition_pseudocounts: jnp.ndarray

class NormalizedEmissionStatistics(NamedTuple):
    normalizer: jnp.ndarray
    normalized_x: jnp.ndarray
    normalized_xxT: jnp.ndarray

# -----------------------------------------------------------------------------

def initial_distribution(params):
    """Return the initial distribution over latent states."""
    return Categorical(probs=params.initial_probs)

def transition_distribution(params, state):
    """Return the distribution of transitioning to states j=1,...,K from state k."""
    return Categorical(probs=params.transition_probs[state])

def emission_distribution(params, state):
    """Return the distribution over emissions given state."""
    return MVNFull(params.emission_means[state], params.emission_covariances[state])

# -----------------------------------------------------------------------------

def log_prior(params, prior_params):
    """Return the log probability of parameters given prior distribution."""
    
    lp = 0.
    
    # Add log probability over prior initial distribution
    prior_initial_distr = Dirichlet(prior_params.initial_probs_conc)
    lp += prior_initial_distr.log_prob(params.initial_probs)

    # Add log probability over prior transition distribution, for each state
    prior_transition_distr = Dirichlet(prior_params.transition_probs_conc)
    lp += prior_transition_distr.log_prob(params.transition_probs).sum()

    # Add log probability over prior emission distribution, for each state
    def _niw_log_prob(cov, mean, prior_loc, prior_conc, prior_df, prior_scale,):
        distr = NormalInverseWishart(prior_loc, prior_conc, prior_df, prior_scale)
        return distr.log_prob(cov, mean)

    lp += vmap(_niw_log_prob)(params.emission_covariances, params.emission_means,
                              prior_params.emission_loc, prior_params.emission_conc,
                              prior_params.emission_df, prior_params.emission_scale,).sum()

    return lp

def log_prob(params, states, emissions):
    """Compute the log joint probabilities of the states and emissions.
    
    Arguments
        params (Parameters):
        states[t,]:
        emissions[t,d]:
    
    Return
        log_joint_prob (float)
    """

    def _step(carry, args):
        lp, prev_state = carry
        this_state, this_emission = args
        lp += transition_distribution(params, prev_state, ).log_prob(this_state)
        lp += emission_distribution(params, this_state, ).log_prob(this_emission)
        return (lp, this_state), None

    # Compute log joint probability of the initial time step
    initial_lp = initial_distribution(params).log_prob(states[0])
    initial_lp += emission_distribution(params, states[0]).log_prob(emissions[0])

    # Compute log joint probability of remaining timesteps via scan
    (lp, _), _ = lax.scan(_step, (initial_lp, states[0]), (states[1:], emissions[1:]))
    return lp

def log_likelihood(params, emissions):
    """Compute log likelihood of emissions at each time step over states.
    
    Arguments
        params (Parameters)
        emissions[t,d]

    Returns
        loglik[t,k]
    """

    # Conditional log likelihood of a single emission, vmapped over all states
    num_states = params.initial_probs.shape[-1]
    _cond_loglik_t = lambda emission: vmap(
        lambda state: emission_distribution(params, state).log_prob(emission)
        )(jnp.arange(num_states))
    
    # Map over all emissions
    return vmap(_cond_loglik_t)(emissions)

def sample(params, num_timesteps, seed):
    """Sample a sequence of latent state states and emissions.

    Arguments
        params (Parameters)
        num_timesteps (int): Number of timesteps to sample
        seed (jr.PRNGKey)

    Returns
        states [num_timesteps,]:
        emissions [num_timesteps, emission_dim]:
    """

    def _step(prev_state, this_seed):
        seed_1, seed_2 = jr.split(this_seed)
        state = transition_distribution(params, prev_state).sample(seed=seed_1)
        emission = emission_distribution(params, prev_state).sample(seed=seed_2)
        return state, (state, emission)    

    # Sample the initial state
    seed_0, seed_1, seed_2 = jr.split(seed, 3)
    initial_state = initial_distribution(params).sample(seed=seed_1)
    initial_emission = emission_distribution(params, initial_state).sample(seed=seed_2)

    # Sample the remaining emissions and states via scan
    next_seeds = jr.split(seed_0, num_timesteps - 1)
    _, (next_states, next_emissions) = lax.scan(_step, initial_state, next_seeds)

    # Concatenate the initial state and emission with the following ones
    states = jnp.concatenate([jnp.expand_dims(initial_state, 0), next_states])
    emissions = jnp.concatenate([jnp.expand_dims(initial_emission, 0), next_emissions])

    return states, emissions