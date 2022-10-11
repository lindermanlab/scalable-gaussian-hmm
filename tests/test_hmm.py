import pytest

from jax import vmap, tree_map
import jax.numpy as jnp
import jax.random as jr
from torch.utils.data import DataLoader
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

def test_suffstats_add_reduce():
    """Test equivalence of sum and reduce methods for suff stats class."""

    nstates = 4
    ndim = 3

    seed_1, seed_2 = jr.split(jr.PRNGKey(1283))

    n_batches = 2
    initial_probs, trans_probs, normd_x, normd_xxT = \
        vmap(partial(random_initialization, nstates=nstates, ndim=ndim)) \
        (jr.split(seed_1, n_batches))

    weights = jr.uniform(seed_2, (n_batches, nstates), minval=0., maxval=1.,)
    marginal_loglik = jnp.array([-340., -20])
    ss0 = NormalizedGaussianHMMSuffStats(
        marginal_loglik=marginal_loglik,
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        weights=weights,
        normd_x=normd_x,
        normd_xxT=normd_xxT,
    )

    ss1 = NormalizedGaussianHMMSuffStats(
        marginal_loglik=marginal_loglik[0],
        initial_probs=initial_probs[0],
        trans_probs=trans_probs[0],
        weights=weights[0],
        normd_x=normd_x[0],
        normd_xxT=normd_xxT[0],
    )

    ss2 = NormalizedGaussianHMMSuffStats(
        marginal_loglik=marginal_loglik[1],
        initial_probs=initial_probs[1],
        trans_probs=trans_probs[1],
        weights=weights[1],
        normd_x=normd_x[1],
        normd_xxT=normd_xxT[1],
    )
    
    ss_reduce = ss0.reduce()
    ss_summed = ss1 + ss2

    assert ss_reduce.marginal_loglik == ss_summed.marginal_loglik == -360.
    assert all(tree_map(lambda a,b: jnp.all(jnp.equal(a,b)), ss_reduce, ss_summed))


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
        marginal_loglik=marginal_loglik,
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        weights=weights,
        normd_x=normd_x,
        normd_xxT=normd_xxT,
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
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=True):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            drop_last=drop_last
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

    # Evaluate
    assert jnp.allclose(refr_lps, test_lps, atol=1e1)
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_covariance_matrices.value, test_hmm.emission_covariance_matrices.value, atol=1e-3)

def test_stochastic_em(n_states=5, n_dim=2, n_steps=1000, n_batches=20, batch_size=4, n_epochs=5):
    """Test equivalence of the stochastic EM algorithm results between this
    GaussianHMM using normalized sufficient stats with the StanfardGaussianHMM."""

    seed_sample, seed_init = jr.split(jr.PRNGKey(9238))
    
    # Make true HMM and generate emissions
    true_hmm_params = make_rnd_hmm_params(n_states, n_dim)
    true_hmm = StandardGaussianHMM(*true_hmm_params)
    _, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))\
                             (jr.split(seed_sample, n_batches), n_steps)
    emissions_loader = ArrayLoader(batch_emissions, batch_size)
    total_emissions = n_batches * n_steps

    # Randomly initialize test and reference GaussianHMM
    refr_hmm = StandardGaussianHMM.random_initialization(seed_init, n_states, n_dim)
    test_hmm = GaussianHMM.random_initialization(seed_init, n_states, n_dim)

    # Run stochastic EM to fit model to data
    
    test_lps = test_hmm.fit_stochastic_em(emissions_loader, total_emissions, num_epochs=n_epochs)
    refr_lps = refr_hmm.fit_stochastic_em(emissions_loader, total_emissions, num_epochs=n_epochs)

    # Evaluate
    assert jnp.allclose(refr_lps, test_lps, atol=1e1)
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_means.value, test_hmm.emission_means.value, atol=1e-3)
    assert jnp.allclose(refr_hmm.emission_covariance_matrices.value, test_hmm.emission_covariance_matrices.value, atol=1e-3)
