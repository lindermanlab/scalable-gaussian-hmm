import jax
from jax import vmap, tree_map
import jax.numpy as jnp
import jax.random as jr
from torch.utils.data import DataLoader
from torch import Generator as torch_rng

from kf import gaussian_hmm

def make_rnd_hmm_params(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_probs = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    return gaussian_hmm.Parameters(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_means=emission_means,
        emission_covariances=emission_covs
    )

def test_em(num_states=3, emission_dim=2, num_timesteps=1000, num_batches=20, num_epochs=400):
    """Test equivalence of the full-batch EM algorithm results between this
    GaussianHMM using normalized sufficient stats with the StanfardGaussianHMM."""

    seed_sample, seed_init = jr.split(jr.PRNGKey(9238))
    
    # Make true HMM and generate emissions
    true_params = make_rnd_hmm_params(num_states, emission_dim)
    _, batch_emissions = vmap(gaussian_hmm.sample, in_axes=(None, None, 0))(
        true_params, num_timesteps, jr.split(seed_sample, num_batches))

    # Randomly initialize Gaussian HMM and fit
    init_params = gaussian_hmm.initialize_model('random', seed_init, num_states, emission_dim)
    prior_params = gaussian_hmm.initialize_prior_from_scalar_values(num_states, emission_dim)
    fitted_params, lps = gaussian_hmm.fit_em(init_params, prior_params, batch_emissions, num_epochs)

    # Evaluate (don't evaluate initial probs, since we only get num_batches observations of it)
    # TODO Write find_optimal_permutation function
    i_perm = jnp.array([1,0,2])
    assert jnp.allclose(true_params.emission_means[i_perm], fitted_params.emission_means, atol=1e-2)
    assert jnp.allclose(true_params.emission_covariances[i_perm][0], fitted_params.emission_covariances[-1], atol=1e-2)
    assert jnp.allclose(true_params.transition_probs.T, fitted_params.transition_probs, atol=1e-2)
    assert jnp.sum(jnp.diff(lps) >= 0.) > 0.8 # log probability should be steadily increasingas

# ------------------------------------------------------------------------------

def _collate(batch):
        """Merges a list of samples to form a batch of tensors."""
        if isinstance(batch[0], jnp.ndarray):
            return jnp.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [_collate(samples) for samples in transposed]
        else:
            return jnp.array(batch)

class ArrayLoader(DataLoader):
    """Generates an iterable over the given array, with option to reshuffle.
    Args:
        dataset (chex.Array or Dataset): Any object implementing __len__ and __getitem__
        batch_size (int): Number of samples to load per batch
        shuffle (bool): If True, reshuffle data at every epoch
        drop_last (bool): If true, drop last incomplete batch if dataset size is
            not divisible by batch size, drop last incomplete batch. Else, keep
            (smaller) last batch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=True, seed=None):
        if seed is not None:
            generator = torch_rng()
            generator.manual_seed(seed)
        else:
            generator = None

        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            drop_last=drop_last,
            generator=generator,
            )

def test_stochastic_em(num_states=3, emission_dim=2, num_timesteps=1000, num_batches=20, batch_size=4, num_epochs=10):
    """Test equivalence of the StEM algorithm results with the full-batch EM results."""

    seed_sample, seed_shuffle, seed_init = jr.split(jr.PRNGKey(9238), 3)
    
    # Make true HMM and generate emissions
    true_params = make_rnd_hmm_params(num_states, emission_dim)
    _, batch_emissions = vmap(gaussian_hmm.sample, in_axes=(None, None, 0))(
        true_params, num_timesteps, jr.split(seed_sample, num_batches))

    # Randomly initialize Gaussian HMM
    init_params = gaussian_hmm.initialize_model('random', seed_init, num_states, emission_dim)
    prior_params = gaussian_hmm.initialize_prior_from_scalar_values(num_states, emission_dim)

    # Test: Fit parameters using stochastic EM. Train on 2x num_epochs
    # Instantiate an emissions generator function for stochastic EM algorithm
    emissions_generator = ArrayLoader(batch_emissions, batch_size, seed=int(seed_shuffle[0]))
    fitted_params, lps = gaussian_hmm.fit_stochastic_em(
        init_params, prior_params, emissions_generator, num_epochs=2*num_epochs)
    
    # Reference: Fit parameters using full-batch EM
    refr_fitted_params, refr_lps = gaussian_hmm.fit_em(init_params, prior_params, batch_emissions, num_epochs)

    # Evaluate emission parameters, trasition parameters, lps
    # Don't compare initial probs, since minibatch will throw that off
    assert jnp.allclose(refr_fitted_params.emission_means, fitted_params.emission_means, atol=1e-2)
    assert jnp.allclose(refr_fitted_params.emission_covariances, fitted_params.emission_covariances, atol=1e-2)
    assert jnp.allclose(refr_fitted_params.transition_probs, fitted_params.transition_probs, atol=1e-2)

    refr_avg_final_lp = refr_lps[-1] / (num_batches * num_timesteps)
    avg_final_lp = lps.ravel()[-1] / (num_batches * num_timesteps)
    assert jnp.allclose(refr_avg_final_lp, avg_final_lp, atol=1e-1)

# -----------------------------------------------------------------------------
def test_parallel_stochastic_em(num_devices=4, num_states=3, emission_dim=2, num_timesteps=500, num_batches=20, batch_size=2, num_epochs=10):
    """Test equivalence of the StEM algorithm results with the full-batch EM results."""

    # Check that the correct number of devices are visible to JAX
    assert jax.local_device_count() == num_devices

    seed_sample, seed_shuffle, seed_init = jr.split(jr.PRNGKey(9238), 3)
    
    # Make true HMM and generate emissions
    true_params = make_rnd_hmm_params(num_states, emission_dim)
    _, batch_emissions = vmap(gaussian_hmm.sample, in_axes=(None, None, 0))(
        true_params, num_timesteps, jr.split(seed_sample, num_batches*num_devices))

    # Randomly initialize Gaussian HMM
    init_params = gaussian_hmm.initialize_model('random', seed_init, num_states, emission_dim)
    prior_params = gaussian_hmm.initialize_prior_from_scalar_values(num_states, emission_dim)

    # Test: Fit parameters using stochastic EM. Train on 2x num_epochs
    # Instantiate an emissions generator function for stochastic EM algorithm
    emissions_generator = ArrayLoader(
        batch_emissions.reshape(num_devices, num_batches, num_timesteps, emission_dim),
        batch_size, seed=int(seed_shuffle[0]))
    fitted_params, lps = gaussian_hmm.fit_stochastic_em(
        init_params, prior_params, emissions_generator, num_epochs=2*num_epochs, parallelize=True)
    
    # Reference: Fit parameters using full-batch EM
    refr_fitted_params, refr_lps = gaussian_hmm.fit_em(init_params, prior_params, batch_emissions, num_epochs)

    # Evaluate emission parameters, trasition parameters, lps
    # Don't compare initial probs, since minibatch will throw that off
    assert jnp.allclose(refr_fitted_params.emission_means, fitted_params.emission_means, atol=1e-2)
    assert jnp.allclose(refr_fitted_params.emission_covariances, fitted_params.emission_covariances, atol=1e-2)
    assert jnp.allclose(refr_fitted_params.transition_probs, fitted_params.transition_probs, atol=1e-2)

    refr_avg_final_lp = refr_lps[-1] / (num_batches * num_timesteps)
    avg_final_lp = lps.ravel()[-1] / (num_batches * num_timesteps)
    assert jnp.allclose(refr_avg_final_lp, avg_final_lp, atol=1e-1)