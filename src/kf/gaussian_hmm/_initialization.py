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

def _kmeans_init(seed, num_states, data, emission_covs_scale=1.,):
    """Initialize GaussianHMM emission parameters from data via k-means algorithm.
    
    Args:
        seed (jr.PRNGKey):
        num_states (int): Number of clusters to fit
        data (jnp.array): Data to fit on, shape (N, emissions_dim)
        emission_covs_scale (float or None): Scale of emission covariances
            initialized to block identity matrices. If None, bootstrap emission
            covariances from kmeans labels. Useful when data is not normalized.
    """
    emissions_dim = data.shape[-1]

    # Set emission means and covariances based on fitted k-means clusters
    kmeans = KMeans(num_states,
                    init='k-means++', n_init=1,
                    random_state=int(seed[-1])).fit(data)
    emission_means = jnp.asarray(kmeans.cluster_centers_)

    # If no covariance scale provided, bootstrap from cluster assignments
    if emission_covs_scale is None:
        labels = kmeans.labels_
        emission_covs = []
        for state in range(num_states):
            _assgns = (labels==state)

            if _assgns.sum() > 1:
                emission_covs.append(jnp.cov(data[_assgns], rowvar=False))
            else: # If states only have 1 assignment, set arbitrary covariance
                emission_covs.append(jnp.eye(emissions_dim))
        emission_covs = jnp.stack(emission_covs)
        
    # Otherwise, set covariance to scaled identity
    else: 
        emission_covs = jnp.tile(
            jnp.eye(emissions_dim) * emission_covs_scale, (num_states, 1, 1))

    return emission_means, emission_covs

def initialize_model(seed, method, num_states, emissions_dim=None, data=None):
    """Initialize a Gaussian HMM via random or k-means initialization.

    Arguments
        seed (jr.PRNGKey)
        method (str): Initialization method, either 'random' or 'kmeans'
        num_states (int): Number of states to initialize 
        emissions_dim (int): Used for 'random' method
        data (jnp.array): Used for 'kmenas' method.
        
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
                        = _kmeans_init(seed_emissions, num_states, data)
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
