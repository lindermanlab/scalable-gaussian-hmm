from functools import partial
from tqdm.auto import trange
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jit
from jax.tree_util import tree_map

from sklearn.cluster import KMeans
import optax

from kf.gaussian_hmm.model import (Parameters, PriorParameters, reduce_gaussian_statistics,
                                   e_step, m_step, log_prior)

__all__ = [
    'initialize_gaussian_hmm',
    'initialize_prior_from_scalar_values',
    'fit_em',
]

def _random_init(seed, num_states, emission_dim):
    """Randomly initialize GaussianHMM emissions parameters."""
   
    emission_means = jr.normal(seed, (num_states, emission_dim))
    emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))

    return emission_means, emission_covs

def _kmeans_init(seed, num_states, emissions_dim, dataloader,
                 step_size=1200, emission_covs_scale=1.,):
    """Initialize GaussianHMM emission parameters from data via k-means algorithm.
    
    Args:
        seed (jr.PRNGKey):
        num_states (int): Number of clusters to fit
        emissions_dim (int): Dimension of emissions
        dataloader (torch.utils.data.Dataloader):
        step_size (int): Number of frames between selected frames of a sequence,
            a larger value results in greater subsampling. Choose large enough
            that we get meaninful data reduction but not so small that kmeans
            fit takes too long. Default: 1200, corresponding to 1 fr/min @ 20 Hz
            (assuming that dataloader sequences are NOT subsampled)
        emission_covs_scale (float or None): Scale of emission covariances
            initialized to block identity matrices. If None, bootstrap emission
            covariances from kmeans labels. TODO
    """
    
    # Get single batch from dataloader and pre-allocate array
    _batch = next(iter(dataloader))
    batch_size, seq_length = _batch.shape[:-1]
    subsampled = onp.empty((len(dataloader), batch_size, seq_length//step_size, emissions_dim))

    # Get data from dataloader, and reshape to (num_samples, emission_dim) array
    for i, batch_emissions in enumerate(dataloader):
        subsampled[i] = batch_emissions[...,::step_size,:]
    subsampled = subsampled.reshape(-1, emissions_dim)

    # Print out some stats
    train_emissions = len(dataloader) * batch_size * seq_length
    print(f'Fitting k-means with {len(subsampled)}/{train_emissions} frames, ' + \
          f'{len(subsampled)/train_emissions*100:.2f}% of training data...' + \
          f'Subsampled at {step_size / 60 / 20:.2f} frames / min.')

    # Set emission means and covariances based on fitted k-means clusters
    kmeans = KMeans(num_states, random_state=int(seed[-1])).fit(subsampled)
    emission_means = jnp.asarray(kmeans.cluster_centers_)

    if emission_covs_scale is None:
        labels = kmeans.labels_
        emission_covs = onp.stack([
            jnp.cov(subsampled[labels==state], rowvar=False) for state in range(num_states)
        ])
    else: 
        emission_covs = jnp.tile(jnp.eye(emissions_dim) * emission_covs_scale, (num_states, 1, 1))

    return emission_means, emission_covs

def initialize_gaussian_hmm(method, seed, num_states, emissions_dim,
                            dataloader=None, step_size=1200):
    """Initialize a Gaussian HMM via random or k-means initialization.

    Arguments
        method (str): Initialization method, either 'random' or 'kmeans'
        seed (jr.PRNGKey)
        num_states (int)
        emissions_dim (int)
        dataloader (torch.utils.data.Dataloader): Training dataset that
            k-means algorithm should fit to. Only used if method == 'kmeans'.
        step_size (int): Training dataset subsampling rate. See description
            in `_kmeans_init`. Only used if method == 'kmeans'.
    
    Return
        Parameters
    """
    
    seed_init, seed_trans, seed_emissions = jr.split(seed, 3)
    
    initial_probs = jr.dirichlet(seed_init, jnp.ones(num_states))
    transition_matrix = jr.dirichlet(seed_trans, jnp.ones(num_states), (num_states,))
    
    if method == 'random':
        emission_means, emission_covs \
                        = _random_init(seed_emissions, num_states, emissions_dim)
    elif method == 'kmeans':
        emission_means, emission_covs \
                        = _kmeans_init(seed_emissions, num_states, emissions_dim,
                                       dataloader, step_size)
    else:
        raise ValueError(f"Expected method to be one of 'random' or 'kmeans', received {method}.")

    return Parameters(
        initial_probs=initial_probs,
        transition_matrix_probs=transition_matrix,
        emission_means=emission_means,
        emission_covariances=emission_covs,
    )

def initialize_prior_from_scalar_values(num_states, emission_dim,
                                        initial_prob_conc=1.1, transition_matrix_conc=1.1,
                                        emission_loc=0., emission_conc=1e-4,
                                        emission_scale=1e-4, emission_extra_df=0.1,):
    """Initialize PriorParameters from scalar values."""
    return PriorParameters(
        initial_prob_conc=initial_prob_conc * jnp.ones(num_states),
        transition_matrix_conc=transition_matrix_conc * jnp.ones((num_states, num_states)),
        emission_loc=emission_loc * jnp.ones((num_states, emission_dim)),
        emission_conc=emission_conc * jnp.ones(num_states),
        emission_scale=emission_scale * jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1)),
        emission_df=(emission_dim + emission_extra_df) * jnp.ones(num_states),
    )
    
        #     emission_loc=emission_loc * jnp.ones(emission_dim),
        # emission_conc=emission_conc,
        # emission_scale=emission_scale * jnp.eye(emission_dim),
        # emission_df=emission_dim + emission_extra_df,

# ==============================================================================
#
# PARAMETER ESTIMATION
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
        batched_latent_stats, batched_emission_stats, batched_lls \
                                    = e_step(params, batched_emissions)

        # TODO Remove once we transition out HiddenMarkovChainStatistics tuple
        batched_initial_stats = batched_latent_stats.initial_probs
        batched_transition_stats = batched_latent_stats.transition_probs

        # Reduce expected statistics along batch dimension
        # latent_stats = tree_map(partial(jnp.sum, axis=0), batched_latent_stats)
        initial_stats = batched_initial_stats.sum(axis=0)
        transition_stats = batched_transition_stats.sum(axis=0)
        emission_stats = reduce_gaussian_statistics(batched_emission_stats, axis=0)

        # Compute MAP estimate
        map_params = m_step(prior_params, initial_stats, transition_stats, emission_stats)
        
        # calculate log likelihood
        lp = log_prior(params, prior_params) + batched_lls.sum()
        return map_params, lp

    log_probs = []
    params = initial_params
    pbar = trange(num_epochs) if verbose else range(num_epochs)
    for _ in pbar:
        params, marginal_loglik = em_step(params)
        log_probs.append(marginal_loglik)
    return params, jnp.array(log_probs)


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
        emissions_generator: An iterable that produces emission minibatches
            of shape [m,t,d]. Automatically shuffles after each epoch.
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
    def minibatch_em_step(params, rolling_stats, minibatch_emissions, learning_rate):
        
        # Compute the sufficient stats given a minibatch of emissions
        this_e_step = partial(e_step, params)
        batched_latent_stats, batched_emission_stats, batched_lls \
                                        = vmap(this_e_step)(minibatch_emissions)

        # Reduce the minibatch statistics along batch dimension, fields have leading shape (K,...)
        minibatch_latent_stats = tree_map(partial(jax.sum, axis=0), batched_latent_stats)
        minibatch_emission_stats = reduce_gaussian_statistics(batched_emission_stats, axis=0)

        # Incoporate minibatch statistics into rolling statistics
        updated_rolling_stats = ...

        # Call M-step
        map_params = m_step(prior_params, *updated_rolling_stats)

        # Calculate expected marginal log likeliood of scaled minibatch
        expected_lp = log_prior(params, prior_params) + num_batches * minibatch_lls.sum()
        
        return map_params, updated_rolling_stats, expected_lp

    # Initialize
    params = initial_params
    rolling_stats = (
        initialize_markov_chain_statistics(num_states),
        initialize_gaussian_statistics(num_states, emission_dim)
    )
    expected_log_probs = jnp.empty((0, len(emissions_generator)))

    # Train
    for epoch in trange(nepochs):
        epoch_expected_lps = []
        for minibatch, minibatch_emissions in enumerate(emissions_generator):
            params, rolling_stats, minibatch_expected_lp = minibatch_em_step(
                params, rolling_stats, minibatch_emissions, learning_rates[epoch][minibatch],
                )
            epoch_expected_lps.append(minibatch_expected_lp)

        # Save epoch mean of expected log probs
        expected_log_probs = jnp.vstack([expected_log_probs, jnp.asarray(epoch_expected_lps)])

    return params, jnp.asarray(expected_log_probs)