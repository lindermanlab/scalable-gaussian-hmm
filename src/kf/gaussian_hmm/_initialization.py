import jax.numpy as jnp
import jax.random as jr
import numpy as onp

from sklearn.cluster import KMeans

from kf.gaussian_hmm._model import (Parameters,
                                   PriorParameters,
                                   NormalizedGaussianHMMStatistics) 


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

def initialize_model(method, seed, num_states, emissions_dim,
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
    transition_probs = jr.dirichlet(seed_trans, jnp.ones(num_states), (num_states,))
    
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
        transition_probs=transition_probs,
        emission_means=emission_means,
        emission_covariances=emission_covs,
    )

# ------------------------------------------------------------------------------

def initialize_prior_from_scalar_values(num_states,
                                        emission_dim,
                                        initial_probs_conc=1.1,
                                        transition_probs_conc=1.1,
                                        emission_loc=0.,
                                        emission_conc=1e-4,
                                        emission_scale=1e-4,
                                        emission_extra_df=0.1,):
    """Initialize PriorParameters from scalar values, with dimension (num_states,)."""
    return PriorParameters(
        initial_probs_conc=initial_probs_conc * jnp.ones(num_states),
        transition_probs_conc=transition_probs_conc * jnp.ones((num_states, num_states)),
        emission_loc=emission_loc * jnp.ones((num_states, emission_dim)),
        emission_conc=emission_conc * jnp.ones(num_states),
        emission_scale=emission_scale * jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1)),
        emission_df=(emission_dim + emission_extra_df) * jnp.ones(num_states),
    )

# ------------------------------------------------------------------------------

def initialize_statistics(num_states, emission_dim, batch_shape=()):
    """Initial GaussianHMM statistics with zero arrays of appropriate shape.
    
    Returns
        chain_stats (HiddenMarkovChainStatistics)
        emission_stats (NormalizedEmissionStatistics)
        normalizer (ndarray)
    """

    stats = NormalizedGaussianHMMStatistics(
        initial_pseudocounts=jnp.zeros((*batch_shape, num_states)),
        transition_pseudocounts=jnp.zeros((*batch_shape, num_states, num_states,)),
        emission_weights=jnp.zeros((*batch_shape, num_states)),
        emission_xxT=jnp.zeros((*batch_shape, num_states, emission_dim, emission_dim)),
        emission_x=jnp.zeros((*batch_shape, num_states, emission_dim)),
    )

    normalizer = jnp.zeros((*batch_shape, num_states))

    return stats, normalizer
