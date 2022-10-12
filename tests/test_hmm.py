import pytest

from jax import vmap, tree_map
import jax.numpy as jnp
import jax.random as jr
from torch.utils.data import DataLoader
from torch import Generator as torch_rng
from functools import partial

from ssm_jax.hmm.models import GaussianHMM as StandardGaussianHMM
from kf import (NormalizedGaussianHMMSuffStats, GaussianHMM)

def random_initialization(seed, nstates, ndim):
    seed_1, seed_2, seed_3 = jr.split(seed, 3)
    initial_probs = jr.dirichlet(seed_1, jnp.ones(nstates))
    transition_matrix = jr.dirichlet(seed_2, jnp.ones(nstates), (nstates,))
    emission_means = jr.normal(seed_3, (nstates, ndim))
    emission_covs = jnp.tile(jnp.eye(ndim), (nstates, 1, 1))
    
    return initial_probs, transition_matrix, emission_means, emission_covs

def test_suffstats_reduce_batch():
    """Reducing a suff stats class with leading dimension (1,) should return itself."""

    nstates = 4
    ndim = 3

    seed_1, seed_2 = jr.split(jr.PRNGKey(1283))

    initial_probs, trans_probs, normd_x, normd_xxT = \
                                random_initialization(seed_1, nstates, ndim)
    weights = jr.uniform(seed_2, (nstates,), minval=0., maxval=1.,)
    marginal_loglik = jnp.array(-120.)
    ss = NormalizedGaussianHMMSuffStats(
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        normd_x=normd_x,
        normd_xxT=normd_xxT,
        weights=weights,
    )

    ss_batch = tree_map(lambda x: jnp.expand_dims(x, axis=0), ss)
    ss_batch_reduced = ss_batch.reduce()

    assert all(tree_map(lambda a,b: jnp.all(jnp.equal(a,b)), ss, ss_batch_reduced))

def make_rnd_hmm_params(nstates=5, ndim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(nstates) / nstates
    transition_matrix = 0.95 * jnp.eye(nstates) + 0.05 * jnp.roll(jnp.eye(nstates), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, nstates + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, nstates + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(ndim), (nstates, 1, 1))

    return initial_probs, transition_matrix, emission_means, emission_covs

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

def test_em(n_states=5, n_dim=2, n_steps=1000, n_batches=20, batch_size=4, n_epochs=5):
    """Test equivalence of the full-batch EM algorithm results between this
    GaussianHMM using normalized sufficient stats with the StanfardGaussianHMM."""

    seed_sample, seed_init = jr.split(jr.PRNGKey(9238))
    
    # Make true HMM and generate emissions
    true_hmm_params = make_rnd_hmm_params(n_states, n_dim)
    true_hmm = StandardGaussianHMM(*true_hmm_params)
    _, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))\
                             (jr.split(seed_sample, n_batches), n_steps)

    # Randomly initialize test and reference GaussianHMM
    refr_hmm = StandardGaussianHMM.random_initialization(seed_init, n_states, n_dim)
    test_hmm = GaussianHMM.random_initialization(seed_init, n_states, n_dim)

    # Run stochastic EM to fit model to data
    test_lps = test_hmm.fit_em(batch_emissions, n_epochs)
    refr_lps = refr_hmm.fit_em(batch_emissions, n_epochs)

    # Evaluate emission parameters, trasition parameters, lps
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_covariance_matrices.value, test_hmm.emission_covariance_matrices.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.initial_probs.value, test_hmm.initial_probs.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.transition_matrix.value, test_hmm.transition_matrix.value, atol=1e-3)
    assert jnp.allclose(refr_lps, test_lps, atol=1e1)

def test_stochastic_em(n_states=5, n_dim=2, n_steps=1000, n_batches=20, batch_size=4, n_epochs=5):
    """Test equivalence of the stochastic EM algorithm results between this
    GaussianHMM using normalized sufficient stats with the StanfardGaussianHMM."""

    seed_sample, seed_shuffle, seed_init = jr.split(jr.PRNGKey(9238), 3)
    
    # Make true HMM and generate emissions
    true_hmm_params = make_rnd_hmm_params(n_states, n_dim)
    true_hmm = StandardGaussianHMM(*true_hmm_params)
    _, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))\
                             (jr.split(seed_sample, n_batches), n_steps)

    # Array loaders MUST have same initial seed in order to release the same
    # batches in the same order
    refr_emissions_loader = ArrayLoader(batch_emissions, batch_size, seed=int(seed_shuffle[0]))
    test_emissions_loader = ArrayLoader(batch_emissions, batch_size, seed=int(seed_shuffle[0]))

    # Randomly initialize test and reference GaussianHMM
    refr_hmm = StandardGaussianHMM.random_initialization(seed_init, n_states, n_dim)
    test_hmm = GaussianHMM.random_initialization(seed_init, n_states, n_dim)

    # Run stochastic EM to fit model to data
    refr_lps = refr_hmm.fit_stochastic_em(refr_emissions_loader, nepochs=n_epochs)
    test_lps = test_hmm.fit_stochastic_em(test_emissions_loader, nepochs=n_epochs)

    # Evaluate emission parameters, trasition parameters, lps
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_covariance_matrices.value, test_hmm.emission_covariance_matrices.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.initial_probs.value, test_hmm.initial_probs.value, atol=1e-2)
    assert jnp.allclose(refr_hmm.transition_matrix.value, test_hmm.transition_matrix.value, atol=1e-2)
    assert jnp.allclose(refr_lps, test_lps, atol=1e1)
