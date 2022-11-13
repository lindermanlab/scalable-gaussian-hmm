import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
from jax.tree_util import tree_map
from functools import partial
import optax
from tqdm.auto import trange, tqdm

from tensorflow_probability.substrates.jax.distributions import Dirichlet
from dynamax.hmm.inference import hmm_smoother, hmm_posterior_mode, compute_transition_probs

from kf.gaussian_hmm._initialization import initialize_statistics
from kf.gaussian_hmm._model import *

__all__ = [
    'e_step',
    'm_step',
    'fit_em',
    'fit_stochastic_em',
    'fit_parallel_stochastic_em',
    'most_likely_states',
]


def _reduce_emission_statistics(stats, axis=0):
    """Compute weighted sum of NormalizedEmissionStatistics along specified axis.

    Args:
        stats with element shapes [b,k], [b,k,d], [b,k,d,d]
    
    Returns
        reduced_stats with element shapes [k,], [k,d], [k,d,d]
    """
    total_weights = stats.normalizer.sum(axis=axis, keepdims=True) # (1,k)
    normd_weights = stats.normalizer/total_weights # (b,k)

    return NormalizedEmissionStatistics(
        normalizer=total_weights.squeeze(), # -> (k,)
        normalized_x=(normd_weights[...,None]*stats.normalized_x).sum(axis=axis), # (k,d)
        normalized_xxT=(normd_weights[...,None, None] * stats.normalized_xxT).sum(axis=axis), # (k,d,d)
    )

def _preduce_emission_statistics(stats):
    """Compute an all-reduce weighted sum of NormalizedEmissionStatistics over
    the pmapped axis. Used in parallelized algorithms.

    Arguments
        stats with element shapes [...,k], [...,k,d], [...,k,d,d] with leading axis 'p'
    
    Returns
        reduced_stats with same shape as stats
    """
    total_weights = lax.psum(stats.normalizer, axis_name='p') # (k,)
    normd_weights = stats.normalizer/total_weights # (k,) with leading axis 'p' 

    return NormalizedEmissionStatistics(
        normalizer=total_weights, # -> (k,)
        normalized_x=lax.psum(normd_weights[...,None] * stats.normalized_x, axis_name='p'), # (k,d)
        normalized_xxT=lax.psum(normd_weights[...,None,None] * stats.normalized_xxT, axis_name='p'), # (k,d,d)
    )

# -----------------------------------------------------------------------------

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

# ==============================================================================
#
# EXPECTATION AND MAXIMIZATION FUNCTIONS
#
# ==============================================================================

def e_step(params, batched_emissions):
    """Compute expected sufficient statistics under the posterior, across all batches.
    
    Arguments
        params (Parameters)
        batched_emissions[b,t,d]

    Returns
        markov_chain_stats (HiddenMarkovChainStatistics([k,...]))
        emission_stats (NormalizedGaussianStatistics([k,...]))
        marginal_loglik (float)
    """

    def _single_e_step(emissions):
        # Run the smoother to calculate the posterior
        posterior = hmm_smoother(params.initial_probs,
                                 params.transition_probs,
                                 log_likelihood(params, emissions))

        # Compute statistics of Hidden Markov Chain
        chain_stats = HiddenMarkovChainStatistics(
            initial_pseudocounts = posterior.initial_probs,
            transition_pseudocounts = compute_transition_probs(params.transition_probs, posterior),
        )

        # Compute normalized weights and emission statistics
        total_weights = jnp.einsum("tk->k", posterior.smoothed_probs)           # shape (K,)
        normd_weights = jnp.where(                                              # shape (T,K)
            total_weights[None,:] > 0., 
            posterior.smoothed_probs / total_weights, 
            0.)
        
        normd_x = jnp.einsum("tk,ti->ki", normd_weights, emissions)
        normd_xxT = jnp.einsum("tk,ti,tj->kij", normd_weights, emissions, emissions)

        emission_stats = NormalizedEmissionStatistics(
            normalizer=total_weights,
            normalized_x=normd_x,
            normalized_xxT=normd_xxT,
        )
        return chain_stats, emission_stats, posterior.marginal_loglik

    # Map the E-step calculations over the batch dimension
    batched_chain_stats, batched_emission_stats, batched_marginal_logliks \
                                        = vmap(_single_e_step)(batched_emissions)

    # Reduce batched outputs along batch dimension
    reduced_chain_stats = tree_map(partial(jnp.sum, axis=0), batched_chain_stats)
    reduced_emission_stats = _reduce_emission_statistics(batched_emission_stats)
    reduced_marginal_logliks = batched_marginal_logliks.sum()

    return reduced_chain_stats, reduced_emission_stats, reduced_marginal_logliks

def m_step(prior_params, markov_chain_stats, emission_stats):
    """Compute MAP estimate of Gaussian HMM parameters.

    Implicitly assumes that num_states > 1.

    Arguments
        prior_params (PriorParameters): Parameter values of prior distributions
        markov_chain_stats (HiddenMarkovChainStatistics([k,...]))
        emission_stats (NormalizedGaussianStatistics([k,...]))
    
    Returns
        map_params (Parameters)
    """

    # Calculate mode of posterior initial distribution
    posterior_initial_conc = (markov_chain_stats.initial_pseudocounts
                              + prior_params.initial_probs_conc)
    initial_probs = Dirichlet(posterior_initial_conc).mode()

    # Calculate mode of posterior transition distribution 
    posterior_transition_conc = (markov_chain_stats.transition_pseudocounts
                                 + prior_params.transition_probs_conc)
    transition_probs = Dirichlet(posterior_transition_conc).mode()

    # Calculate mode of posterior emission distribution
    def _single_emission_m_step(normd_stats, prior_niw_mean_params):
        # Convert prior NIW parameters to natural parameterization
        natural_prior_params = niw_convert_mean_to_natural(*prior_niw_mean_params)
        
        # Normalize prior parameters by emission stats normalizer
        normd_natural_prior_params = tree_map(
            lambda eta: eta / normd_stats.normalizer, natural_prior_params
        )

        # Compute posterior parameters
        normd_natural_posterior_params = tree_map(
            jnp.add,
            normd_natural_prior_params,
            (1, normd_stats.normalized_xxT, normd_stats.normalized_x, 1)
        )
        
        # Convert natural posterior parameters to mean parameterization
        posterior_loc, _, _, posterior_scale \
            = niw_convert_natural_to_mean(*normd_natural_posterior_params)

        # Return modal values of posterior distribution
        modal_cov = posterior_scale / normd_natural_posterior_params[0]
        modal_mean = posterior_loc
        return modal_cov, modal_mean

    # Map the emissions M-step calculation over the state dimension
    covs, means = vmap(_single_emission_m_step)(
        emission_stats,
        (prior_params.emission_loc, prior_params.emission_conc, prior_params.emission_df, prior_params.emission_scale),
    )

    return Parameters(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_means=means,
        emission_covariances=covs,
    )
    
# ==============================================================================
#
# FULL-BATCH EM ALGORITHM
#
# ==============================================================================

def fit_em(initial_params, prior_params, batched_emissions, num_epochs=50, verbose=True):
    """Estimate model parameters from emissions using Expectation-Maximization (EM).
    
    Arguments
        initial_params (Parameters)
        prior_params (PriorParams)
        batch_emissions[b,t,d]
        num_epochs (int): Number of EM iterations to run over full dataset
        verbose (bool): If true, print progress bar.
    
    Returns
        fitted_params (Parameters)
        lps[num_epochs,]
    """

    @jit
    def em_step(params):
        # Compute expected sufficient statistics
        markov_chain_stats, emission_stats, lls = e_step(params, batched_emissions)

        # Compute MAP estimate
        map_params = m_step(prior_params, markov_chain_stats, emission_stats)
        
        # calculate log likelihood
        lp = log_prior(params, prior_params) + lls
        return map_params, lp

    log_probs = []
    params = initial_params
    pbar = trange(num_epochs) if verbose else range(num_epochs)
    for _ in pbar:
        params, marginal_loglik = em_step(params)
        log_probs.append(marginal_loglik)
    return params, jnp.array(log_probs)

# ==============================================================================
#
# STOCHASTIC EM ALGORITHM
#
# ==============================================================================

def _update_rolling_emission_statistics(rolling_stats, minibatch_stats, alpha, scale):
    """Update rolling set of emission stats with stats from minibatch.
    
    Used in stochastic EM algorithm.

    Arguments
        rolling_stats (NormalizedEmissionStats([k,...])):
            Emission stats representing accumulation of previously-seen emissions
        minibatch_stats (NormalizedEmissionStats([k,...])): 
            Emission stats calculated from this minibatch of emissions
        alpha (float): Learning rate
        scale (float): Factor to scale minibatch_stats up by.
            TODO When using normalization by len(batch_emissions), we should
            then be able to omit this term.

    Returns
       updated_rolling_stats (NormalizedEmissionStats([k,...])):
    """

    rolling_normalizer = (1-alpha) * rolling_stats.normalizer
    minibatch_normalizer = alpha * scale * minibatch_stats.normalizer
    updated_normalizer = rolling_normalizer + minibatch_normalizer

    _weighted_update = (lambda rolling_stat, minibatch_stat:
        jnp.einsum('k,k...->k...', rolling_normalizer/updated_normalizer, rolling_stat)
        + jnp.einsum('k,k...->k...', minibatch_normalizer/updated_normalizer, minibatch_stat)
    )
    
    return NormalizedEmissionStatistics(
        normalizer=updated_normalizer,
        normalized_x=_weighted_update(rolling_stats.normalized_x, minibatch_stats.normalized_x),
        normalized_xxT=_weighted_update(rolling_stats.normalized_xxT, minibatch_stats.normalized_xxT),    
    )

def fit_stochastic_em(initial_params, prior_params, emissions_generator,
                      schedule=None, num_epochs=5, verbose=True):
    """Estimate model parameters from emissions using stochastic Expectation-Maximization (StEM).

    Let the original dataset consists of N independent sequences of length T.
    The StEM algorithm then performs EM on each random subset of M sequences
    (not timesteps) during each epoch. Specifically, it will perform N//M
    iterations of EM per epoch. The algorithm uses a learning rate schedule
    to anneal the minibatch sufficient statistics at each stage of training.
    If a schedule is not specified, an exponentially decaying model is used
    such that the learning rate which decreases by 5% at each epoch.

    NB: This algorithm assumes that the `emissions_generator` object automatically
    shuffles minibatch sequences before each epoch. It is up to the user to
    correctly instantiate this object to exhibit this property. For example,
    `torch.utils.data.DataLoader` objects implement such a functionality.
    
    Arguments
        initial_params (Parameters)
        prior_params (PriorParams)
        emissions_generator (Generator->[m,t,d]): Produces minibatches of
            emissions. Automatically shuffles after each epoch.
        schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
            schedule; defaults to exponential schedule.
        num_epochs (int): Number of StEM iterations to run over full dataset
        verbose (bool): If true, print progress bar.
    
    Returns
        fitted_params (Parameters)
        log_probs [num_epochs, num_batches]
    """
    
    num_batches = len(emissions_generator)
    num_states = initial_params.initial_probs.shape[-1]
    emission_dim = initial_params.emission_means.shape[-1]

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

    @jit
    def _minibatch_em_step(params, rolling_stats, minibatch_emissions, learning_rate):
        # minibatch_emissions[m,t,d]
        rolling_markov_chain_stats, rolling_emission_stats = rolling_stats

        # Compute the sufficient stats given a minibatch of emissions
        minibatch_markov_chain_stats, minibatch_emission_stats, minibatch_lls \
                                        = e_step(params, minibatch_emissions)

        # Convexly combine rolling statistics with minibatch statistics
        updated_markov_chain_stats = tree_map(
            lambda s0, s1: (1-learning_rate) * s0 + learning_rate * num_batches * s1,
            rolling_markov_chain_stats, minibatch_markov_chain_stats
        )

        updated_emission_stats = _update_rolling_emission_statistics(
            rolling_emission_stats, minibatch_emission_stats, learning_rate, num_batches,
        )

        # Call M-step
        map_params = m_step(prior_params, updated_markov_chain_stats, updated_emission_stats)

        # Calculate expected marginal log likeliood of scaled minibatch
        expected_lp = log_prior(params, prior_params) + num_batches * minibatch_lls
        
        return map_params, (updated_markov_chain_stats, updated_emission_stats), expected_lp

    # Initialize
    params = initial_params
    rolling_stats = initialize_statistics(num_states, emission_dim)
    expected_log_probs = jnp.empty((0, len(emissions_generator)))

    # Train
    for epoch in trange(num_epochs):
        epoch_expected_lps = []
        for minibatch, minibatch_emissions in enumerate(emissions_generator):
            params, rolling_stats, minibatch_expected_lp = _minibatch_em_step(
                params, rolling_stats, minibatch_emissions, learning_rates[epoch][minibatch],
                )
            epoch_expected_lps.append(minibatch_expected_lp)

        # Save epoch mean of expected log probs
        expected_log_probs = jnp.vstack([expected_log_probs, jnp.asarray(epoch_expected_lps)])

    return params, jnp.asarray(expected_log_probs)

# ==============================================================================
#
# PARALLELIZED STOCHASTIC EM
#
# ==============================================================================    

def fit_parallel_stochastic_em(initial_params, prior_params, emissions_generator,
                               schedule=None, num_epochs=5, verbose=True):
    """Estimate model parameterss from emissions using stochastic
    Expectation-Maximization (StEM) parallelized across multiple devices.

    This algorithm makes use of JAX's pmap and collective all-reduce operations
    to perform the expensive E-steps in parallel. It assumes that there are 'p'
    devices, which must EXACTLY match the 'p' in the minibatch shape returned
    by the emissions_generator iterable. All other arguments are the same.

    Currently, this code assumes parallelization over multiple CPU cores on the
    same devices. This requires the user to set the environment flags
        XLA_FLAGS=--xla_force_host_platform_device_count=[p]

    For details regarding the stochastic EM algorithm, see documentation for
    `fit_stochastic_em` code above.
    
    Arguments
        initial_params (Parameters)
        prior_params (PriorParams)
        emissions_generator(Generator->[p,m,t,d]): Produces minibatches of
            emissions. Automatically shuffles after each epoch.
        schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
            schedule; defaults to exponential schedule.
        num_epochs (int): Number of StEM iterations to run over full dataset
        verbose (bool): If true, print progress bar. TODO
    
    Returns
        fitted_params (Parameters)
        log_probs[num_epochs, num_batches]
    """
    
    num_batches = len(emissions_generator)
    num_states = initial_params.initial_probs.shape[-1]
    emission_dim = initial_params.emission_means.shape[-1]

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

    @partial(pmap, in_axes=(None, None, None, None, None, 0), axis_name='p')
    def em_step(prior_params, params, rolling_stats, learning_rate, num_batches, local_batch_emissions):
        # INPUT: local_batch_emissions[m,t,d] varying across outer axis 'p'
        # OUTPUT: NamedTuples with leading axis [k,...], varying across outer axis 'p'
        local_markov_chain_stats, local_emission_stats, local_lls \
                                        = e_step(params, local_batch_emissions)
        
        # Reduce batch statistics across outer batch axis 'p'
        # OUTPUT: NamedTuples have leading axis [k,...], identical across outer axis 'p'
        collective_markov_chain_stats, collective_lls = tree_map(
            partial(lax.psum, axis_name='p'), (local_markov_chain_stats, local_lls)
        )
        collective_emission_stats = _preduce_emission_statistics(local_emission_stats)

        # Convexly combine rolling statistics with collective statistics
        # All stats and lls are now identical across devices (across outer axis 'p')
        rolling_markov_chain_stats, rolling_emission_stats = rolling_stats
        
        updated_markov_chain_stats = tree_map(
            lambda s0, s1: (1-learning_rate) * s0 + learning_rate * num_batches * s1,
            rolling_markov_chain_stats, collective_markov_chain_stats
        )

        updated_emission_stats = _update_rolling_emission_statistics(
            rolling_emission_stats, collective_emission_stats, learning_rate, num_batches,
        )

        # Call M-step
        map_params = m_step(prior_params, updated_markov_chain_stats, updated_emission_stats)

        # Calculate expected marginal log likeliood of scaled minibatch
        expected_lp = log_prior(params, prior_params) + num_batches * collective_lls

        return map_params, (updated_markov_chain_stats, updated_emission_stats), expected_lp


    # Initialize and train
    params = initial_params
    rolling_stats = initialize_statistics(num_states, emission_dim)
    expected_log_probs = jnp.empty((0, len(emissions_generator)))

    for epoch in range(num_epochs):
        epoch_expected_lps = []

        pbar = (
            enumerate(tqdm(emissions_generator, desc=f'epoch {epoch}'))
            if verbose else enumerate(emissions_generator)
        )
        for minibatch, minibatch_emissions in pbar:
            # PMAPPED INPUT into em_step: minibatch_emissions[p,m,t,d]
            # OUTPUTS: NamedTuples whose leaves have leading axis [p,...],
            #          values are identical across this dimension
            pbatch_params, pbatch_stats, pbatch_expected_lps \
                                                = em_step(prior_params,
                                                          params,
                                                          rolling_stats,
                                                          learning_rates[epoch][minibatch],
                                                          num_batches,
                                                          minibatch_emissions)

            # Select one of these (redundant) values
            params, rolling_stats, minibatch_expected_lp = tree_map(
                lambda arr: arr[0], (pbatch_params, pbatch_stats, pbatch_expected_lps)
            )
            
            epoch_expected_lps.append(minibatch_expected_lp)

        # Save epoch mean of expected log probs
        expected_log_probs = jnp.vstack([expected_log_probs, jnp.asarray(epoch_expected_lps)])

    return params, jnp.asarray(expected_log_probs)

# ==============================================================================
#
# VITERBI PATH
#
# ==============================================================================

def most_likely_states(params, emissions):
    """Compute Viterbi path for observed emissions."""

    return  hmm_posterior_mode(params.initial_probs,
                               params.transition_probs,
                               log_likelihood(params, emissions))