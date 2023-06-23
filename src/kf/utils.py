from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array, Float

@jit
def kl_gaussian(mu_0: Float[Array, "D"],
                cov_0: Float[Array, "D D"],
                mu_1: Float[Array, "D"],
                cov_1: Float[Array, "D D"],):
    """Calculate KL divergence between two multivariate Gaussian distributions.
    
    Specifically, calculates KL(p_1 || p_0).
    """
    dim = mu_0.shape[-1]
    
    dist = jnp.trace(jnp.linalg.solve(cov_1, cov_0))
    dist += (mu_1-mu_0).T @ jnp.linalg.solve(cov_1, mu_1-mu_0)
    dist += jnp.log(jnp.linalg.det(cov_1)) - jnp.log(jnp.linalg.det(cov_0))
    dist -= dim
    
    return 0.5 * dist

def pdist_symmetric_kl(mus: Float[Array, "N D"], covs: Float[Array, "N D D"],):
    """Compute the symmetric KL divergence between pairs of Gaussian parameters."""
    pdist_fn = vmap(vmap(kl_gaussian, (0,0,None,None)), (None,None,0,0))
    kl = pdist_fn(mus, covs, mus, covs)
    return 0.5 * (kl + kl.T)

def get_redundant_states(dist_mat: Float[Array, "N N"], atol: float=1e-3, verbose: bool=False):
    """Return indices of redundant states based on calculate pairwise distance matrix."""

    # Mask the lower triangle and diagonal with NaNs to avoid double-counting distances
    mask = jnp.tril(jnp.zeros_like(dist_mat) * jnp.nan)
    masked_dist_mat = dist_mat + mask

    redundant_states_x, redundant_states_y = jnp.nonzero(jnp.isclose(masked_dist_mat, 0, atol=atol))
    if verbose:
        print('Found the following redundant state coordinates:')
        print(redundant_states_x, redundant_states_y)
        
    # Loop through redundant_states
    all_redundant_states = []
    while len(redundant_states_x) > 0:
        this_state = redundant_states_x[0]
        mask_x = (redundant_states_x == this_state)

        # Filter out states which are redundant to `this_state`
        states_matching_this_state = redundant_states_y[mask_x]
        all_redundant_states.append(states_matching_this_state)
        if verbose:
            print(f'Found states {states_matching_this_state} to be redundant with state {this_state}.')
        
        # Accounting: Processed everything associated with `this_state`
        redundant_states_x = redundant_states_x[~mask_x]
        redundant_states_y = redundant_states_y[~mask_x]
        if verbose:
            print(f'Removed repeats of `this_state`={this_state}: ', end='')
            print(redundant_states_x, redundant_states_y)
        
        # Accounting: Remove states that are already marked for replacement
        mask_y = jnp.isin(redundant_states_x, states_matching_this_state)
        redundant_states_x = redundant_states_x[~mask_y]
        redundant_states_y = redundant_states_y[~mask_y]
        if verbose:
            print(f'Removed reundant states: ', end='')
            print(redundant_states_x, redundant_states_y)
            print('')

    return jnp.concatenate(all_redundant_states)

def quick_kmeans_plusplus(seed: jr.PRNGKey,
                          n_addtl_clusters: int,
                          existing_centroids: Float[Array, "M D"],
                          data: Float[Array, "N D"],):
    """Initialize `n_addtl_clusters` cluster centroids from data using k-means++.

    Quick (and dirty) because this algorithm ensures that new cluster centroids
    are distant from M existing centroids, but does NOT check that new cluster
    centroids are distant from each other.
    """

    # Calculate squared Euclidean distances between data points and existing centers
    dists = jnp.linalg.norm(data[:,None,:] - existing_centroids[None,:,:], axis=-1)**2
    cum_dists = dists.sum(axis=-1)
    
    # Normalize distances to fall between 0 and 1
    normd_dists = cum_dists / cum_dists.max()

    # Choose candidates by sampling with probabiity proportional to distance
    i_candidates = jr.choice(seed, (n_addtl_clusters,), replace=False, p=normd_dists)

    return data[i_candidates, :]