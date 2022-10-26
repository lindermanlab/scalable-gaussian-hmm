"""Code for performing inference in Gaussian HMMs using normalized sufficient
statistics and an inverse Wishart prior.

Indirectly inherits heavily from the Dynamax codebase, to which this codebase was 
once tied much more closely. The main changes here are breaking out of the class
structure to ensure compatibility with jax and lax parallelization functions.
"""

from collections import namedtuple
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from jax.tree_util import tree_map

from tensorflow_probability.substrates.jax.distributions import (
    Dirichlet, Categorical, MultivariateNormalFullCovariance as MVNFull)
from dynamax.distributions import NormalInverseWishart

from dynamax.hmm.inference import hmm_smoother, hmm_posterior_mode, compute_transition_probs

__all__ = [
    'Parameters',
    'PriorParameters',
    'log_prob',
    'conditional_log_likelihood',
    'most_likely_states',
    'sample',
    'e_step',
    'm_step',
]

# =============================================================================
#
# USEFUL CONTAINERS
#
# =============================================================================

Parameters = namedtuple(
    'Parameters',
    ['initial_probs', 'transition_matrix_probs', 'emission_means', 'emission_covariances',])

PriorParameters = namedtuple(
    'PriorParameters',
    ['initial_prob_conc', 'transition_matrix_conc',
     'emission_loc', 'emission_conc', 'emission_scale', 'emission_extra_df',],
    defaults=[1.1, 1.1, 0.0, 1e-4, 1e-4, 0.1]
)

# TODO Remove this and just break out the values. This is superfluous.
HiddenMarkovChainStatistics = namedtuple(
    'HiddenMarkovChainStatistics',
    ['initial_probs', 'transition_probs']
)

def initialize_markov_chain_statistics(num_states, batch_shape=()):
    """Return HiddenMarkovChainStatistics initialized with shaped zero arrays."""

    return HiddenMarkovChainStatistics(
        initial_probs=jnp.zeros((*batch_shape, K)),
        transition_probs=jnp.zeros((*batch_shape, K, K,)),
    )

NormalizedGaussianStatistics = namedtuple(
    'NormalizedGaussianStatistics',
    ['normalized_x', 'normalized_xxT', 'normalizer'])

def initialize_gaussian_statistics(num_states, emissions_dim, batch_shape=()):
    """Return NormalizedGaussianStatistics initialized with shaped zero arrays."""
    
    return NormalizedGaussianStatistics(
        normalized_x=jnp.zeros((*batch_shape, num_states, emissions_dim)),
        normalized_xxT=jnp.zeros((*batch_shape, num_states, emissions_dim, emissions_dim)),
        normalizer=jnp.zeros((*batch_shape, num_states)),
    )

def reduce_gaussian_statistics(stats, axis=0):
    """Reduce NormalizedGaussianSufficientStatistics along specified axis."""
    total_weights = stats.normalizer.sum(axis=axis, keepdims=True)
    normd_weights = stats.normalizer/total_weights

    return NormalizedGaussianStatistics(
        normalized_x=(normd_weights[...,None]*stats.normalized_x).sum(axis=axis),
        normalized_xxT=(normd_weights[...,None, None] * stats.normalized_xxT).sum(axis=axis),
        normalizer=total_weights.squeeze()
    )

# =============================================================================
#
# NormalInverseWishart helper functions
#
# =============================================================================

def niw_convert_mean_to_natural(loc, conc, df, scale):
    """Convert NIW mean parameters to natural parameters."""
    dim = loc.shape[-1]
    eta_1 = df + dim + 2
    eta_2 = scale + conc * jnp.outer(loc, loc)
    eta_3 = conc * loc
    eta_4 = conc
    return eta_1, eta_2, eta_3, eta_4

def niw_convert_natural_to_mean(eta_1, eta_2, eta_3, eta_4):
    """Convert NIW natural parameters to mean parameters."""
    dim = eta_3.shape[-1]
    loc = eta_3 / eta_4
    conc = eta_4            
    scale = eta_2 - jnp.outer(eta_3, eta_3) / eta_4
    df = eta_1 - dim - 2
    return loc, conc, df, scale

def niw_posterior_mode(stats, prior_params):
    """Compute the posterior NIW mode given Gaussian statistics and prior parameters.

    Operates on unbatched NormalizedGaussianStatistics. If batched, call this 
    function with vmap over the batch dimension(s).
    
    Arguments
        normd_stats (NormalizedGaussianStatistics)
        prior_params (PriorParameters)

    Returns
        modal_covariance[d,d]
        modal_mean[d]
    """
    
    # Convert prior parameters to natural parameterization
    emission_dim = stats.normalized_x.shape[-1]
    natural_prior_params = niw_convert_mean_to_natural(
        prior_params.emission_loc, prior_params.emission_conc,
        emission_dim + prior_params.emission_extra_df, prior_params.emission_scale)

    # Normalize prior natural parameters by the normalizer of the sufficient statistics
    normd_natural_prior_params = natural_prior_params / stats.normalizer

    # Compute posterior by adding normalized natural parameters of prior and likelihood    
    natural_posterior_params = tree_map(
        jnp.add, normd_natural_prior_params, (1., stats.normalized_xxT, stats.normalized_x, 1.)
    )

    # Convert natural posterior parameters to mean parameterization
    posterior_loc, _, _, posterior_scale \
        = niw_convert_natural_to_mean(*natural_posterior_params)

    # Compute mode
    modal_covariance = posterior_scale / normd_natural_prior_params[0]
    modal_mean = posterior_loc

    return modal_covariance, modal_mean

# =============================================================================
#
# GAUSSIAN HIDDEN MARKOV MODEL
#
# =============================================================================

def initial_distribution(params: Parameters):
    """Return the initial distribution over latent states."""
    return Categorical(probs=params.initial_probs)

def transition_distribution(params: Parameters, state: int):
    """Return the distribution of transitioning to states j=1,...,K from state k."""
    return Categorical(probs=params.transition_matrix_probs[state])

def emission_distribution(params: Parameters, state: int):
    """Return the distribution over emissions given state."""
    return MVNFull(params.emission_means[state], params.emission_covariances[state])

# -----------------------------------------------------------------------------

def log_prior(params, prior_params):
    """Return the log probability of parameters given prior distribution.
    
    Arguments
        params (Parameters)
        prior_params (PriorParameters)

    Returns
        log_likelihood_under_prior (float)
    """
    
    K, D = params.initial_probs.shape[-1], params.emission_means.shape[-1]

    lp = 0.
    
    # Add log probability over prior initial distribution
    prior_initial_distr = Dirichlet(jnp.ones(K) * prior_params.initial_prob_conc)
    lp += prior_initial_distr.log_prob(params.initial_probs)

    # Add log probability over prior transition distribution, for each state
    prior_transition_distr = Dirichlet(jnp.ones(K) * prior_params.transition_matrix_conc)
    lp += prior_transition_distr.log_prob(params.transition_matrix_probs).sum()

    # Add log probability over prior emission distribution, for each state
    prior_emission_distr = NormalInverseWishart(
        prior_params.emission_loc, prior_params.emission_conc,
        prior_params.emission_extra_df + D, prior_params.emission_scale)  
    lp += prior_emission_distr.log_prob(params.emission_covariances, params.emission_means).sum()

    return lp

def log_prob(params: Parameters, states: jnp.ndarray, emissions: jnp.ndarray):
    """Compute the log joint probabilities of the states and emissions.
    
    Arguments
        params (Parameters):
        states[t,]
        emissions[t,d]
    
    Return
        log_joint_prob (float)
    """

    def _step(carry, args):
        lp, prev_state = carry
        this_state, this_emission = args
        lp += transition_distribution(params, prev_state, ).log_prob(this_state)
        lp += emission_distribution(params, this_state, ).log_prob(this_emission)
        return (lp, state), None

    # Compute log joint probability of the initial time step
    initial_lp = initial_distribution(params).log_prob(states[0])
    initial_lp += emission_distribution(params, states[0]).log_prob(emissions[0])

    # Compute log joint probability of remaining timesteps via scan
    (lp, _), _ = lax.scan(_step, (initial_lp, states[0]), (states[1:], emissions[1:]))
    return lp

def conditional_log_likelihood(params, emissions):
    """Compute log likelihood of emissions at each time step over states.
    
    Arguments
        params (Parameters)
        emissions[t,d]

    Returns
        conditional_loglik[t,k]
    """

    # Conditional log likelihood of a single emission, vmapped over all states
    num_states = params.initial_probs.shape[-1]
    _cond_loglik_t = lambda emission: vmap(
        lambda state: emission_distribution(params, state).log_prob(emission)
        )(jnp.arange(num_states))
    
    # Map over all emissions
    return vmap(_cond_loglik_t)(emissions)

# -----------------------------------------------------------------------------

def most_likely_states(params, emissions):
    """Compute Viterbi path.
    
    Arguments
        params (Parameters)
        emissions[t,d]
    
    Returns
        most_likely_states[t,]
    """

    return  hmm_posterior_mode(params.initial_probs,
                               params.transition_matrix_probs,
                               conditional_log_likelihood(params, emissions))

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
# -----------------------------------------------------------------------------

def e_step(params: Parameters, batched_emissions: jnp.ndarray):
    """Compute expected sufficient statistics under the posterior.
    
    Arguments
        params (Paramters)
        batched_emissions[b,t,d]

    Returns
        batched_latent_stats: HiddenMarkovChainStatistics with leading (B,K,...) dimensions
        batched_emission_stats: NormalizedGaussianStatistics with leading (B,K,...) dimensions
        posterior_marginal_loglik[b,]
    """

    def _single_e_step(emissions):
        # Run the smoother to calculate the posterior
        posterior = hmm_smoother(params.initial_probs,
                                 params.transition_matrix_probs,
                                 conditional_log_likelihood(params, emissions))

        # Compute statistics of Hidden Markov Chain
        latent_stats = HiddenMarkovChainStatistics(
            initial_probs = posterior.initial_probs,
            transition_probs = compute_transition_probs(params.transition_matrix_probs, posterior),
        )

        # Compute normalized weights and emission statistics
        total_weights = jnp.einsum("tk->k", posterior.smoothed_probs)           # shape (K,)
        normd_weights = jnp.where(                                              # shape (T,K)
            total_weights[None,:] > 0., 
            posterior.smoothed_probs / total_weights, 
            0.)
        
        normd_x = jnp.einsum("tk,ti->ki", normd_weights, emissions)
        normd_xxT = jnp.einsum("tk,ti,tj->kij", normd_weights, emissions, emissions)

        emission_stats = NormalizedGaussianStatistics(
            normalized_x=normd_x,
            normalized_xxT=normd_xxT,
            normalizer=total_weights,
        )
        return latent_stats, emission_stats, posterior.marginal_loglik

    # Map the E-step calculations over the batch dimension
    return vmap(_single_e_step)(batched_emissions)

def m_step(prior_params, initial_stats, transition_stats, emission_stats):
    """Compute MAP estimate of Gaussian HMM parameters.

    Implicitly assumes that num_states > 1.

    Arguments
        initial_stats[k,]
        transition_stats[k,k]
        emission_stats (NormalizedGaussianStatistics): Normalized
            Gaussian statistics, with leading (K,...) dimensions
        prior_params (PriorParameters): Parameter values of prior distributions
    
    Returns
        Parameters: Maximum a posterior (MAP) parameters of Gaussian HMM
    """

    # Calculate mode of posterior initial distribution
    postr_initial_conc = initial_stats + prior_params.initial_prob_conc
    postr_initial_distr = Dirichlet(postr_initial_conc)
    initial_probs = postr_initial_distr.mode()

    # Calculate mode of posterior transition distribution 
    postr_transition_conc = transition_stats + prior_params.transition_matrix_conc
    postr_transition_distr = Dirichlet(postr_transition_conc)
    transition_matrix_probs = postr_transition_distr.mode()

    # Map the emissions M-step calculation over the state dimension
    covs, means = vmap(niw_posterior_mode, in_axes=(0, None))(emission_stats, prior_params)

    return Parameters(
        initial_probs=initial_probs,
        transition_matrix_probs=transition_matrix_probs,
        emission_means=means,
        emission_covariances=covs,
    )