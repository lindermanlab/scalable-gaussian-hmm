"""Test EM steps with new ssm-jax fork."""

import pytest

import jax.config
jax.config.update('jax_platform_name', 'cpu')

from functools import partial, reduce
import jax
import jax.numpy as jnp
import jax.random as jr

from kf.inference import streaming_parallel_e_step

from ssm_jax.hmm.models import GaussianHMM
from ._standard_em import standard_em_step

# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -----------------------------------------------------------------------------

def allclose_errmsg(test_arr, ref_arr, rtol=1e-5, atol=1e-8, title=''):
    err = jnp.abs(test_arr-ref_arr)
    tol = atol + rtol * jnp.abs(ref_arr)

    isntclose = err > tol
    if err[isntclose].size > 0:
        argmax_isntclose = jnp.argmax(err[isntclose])
        max_isntclose = err[isntclose][argmax_isntclose]
        tol_at_max = tol[isntclose][argmax_isntclose]

        msg = f'{title} (atol={atol:.0e}, rtol={rtol:.0e}): ' \
              + f'max error {max_isntclose:.2e} > {tol_at_max:.2e}'
    else:
        argmax_isclose = jnp.argmax(err)
        max_isclose = err.ravel()[argmax_isclose]
        tol_at_max  = tol.ravel()[argmax_isclose]

        msg = f'{title} (atol={atol:.0e}, rtol={rtol:.0e}): ' \
              + f'max error {max_isclose:.2e} <= {tol_at_max:.2e}'
    
    return msg

def assert_allclose_with_errmsg(test_arr, ref_arr, rtol=1e-5, atol=1e-8, title=''):
    """Assert 2 arrays are all close, else print useful error message."""
    msg = allclose_errmsg(test_arr, ref_arr, rtol, atol, title)

    assert jnp.allclose(test_arr, ref_arr, rtol, atol), msg
    
def get_leading_dim(ss) -> jnp.ndarray:
    return jnp.array([len(ss[k]) for k in ss.__dataclass_fields__.keys()])

class SimpleDataloader():
    def __init__(self, emissions, num_devices, num_batches_per_device, num_timesteps_per_batch):
        self.emissions_dim = emissions.shape[-1]
        self._emissions = emissions.reshape(-1, self.emissions_dim)

        self.num_devices = num_devices
        self.num_batches_per_device = num_batches_per_device
        self.num_timesteps_per_batch = num_timesteps_per_batch

    def __len__(self):
        return (
            len(self._emissions)
            // (self.num_devices * self.num_batches_per_device * self.num_timesteps_per_batch)
        )
    
    @property
    def batch_shape(self):
        return (self.num_devices,
                self.num_batches_per_device,
                self.num_timesteps_per_batch,
                self.emissions_dim)

    def __iter__(self):
        self.emissions = self._emissions.reshape(len(self), *self.batch_shape)
        self._iter_count = 0
        return self

    def __next__(self):        
        if self._iter_count >= len(self):
            raise StopIteration
        
        self._iter_count += 1

        return self.emissions[self._iter_count-1]
        
# -----------------------------------------------------------------------------

def make_rnd_hmm(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    # Make a true HMM
    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

    return true_hmm

def make_rnd_hmm_and_data(num_states=5, emission_dim=2, num_timesteps=2000):
    """Returns emissions with batch axis, shape: (1, num_timesteps, emission_dim)
    and the GaussianHMM and latent states which generated the emissions."""
    true_hmm = make_rnd_hmm(num_states, emission_dim)
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)
    batch_emissions = emissions[None, ...]
    return true_hmm, true_states, batch_emissions

# -----------------------------------------------------------------------------

def test_streaming_pmap():
    """All batches have the same number of obs, i.e. (M, T_M, D) for M batches
    and T_M = num_obs // M."""
    
    # -------------------------------------------------------------------------
    # Setup: Generate emissions, and ref and test hmms
    # -------------------------------------------------------------------------
    num_states = 5
    emission_dim = 2
    num_timesteps = 2000

    # Generate (unbatched) emissions. This will be used by ref_hmm
    _, _, emissions = make_rnd_hmm_and_data(num_states, emission_dim, num_timesteps)

    # Randomly initialize a HMM. These are the same because seeded with same key    
    ref_hmm = GaussianHMM.random_initialization(jr.PRNGKey(1), 2*num_states, emission_dim)
    test_hmm = GaussianHMM.random_initialization(jr.PRNGKey(1), 2*num_states, emission_dim)

    # -------------------------------------------------------------------------
    # Compare: Generate emissions, and ref and test hmms
    # -------------------------------------------------------------------------
    num_iters = 5

    # Reference point: Full-batch code. Updates hmm out-of-place
    for _ in range(num_iters):
        ref_hmm, ref_nss = standard_em_step(ref_hmm, emissions)

    # -----------------------------------------------------------
    # Test code: Streaming batches emissions code. Updates hmm in-place

    # e-step reshapes obs into (num_devices, num_batches_per_device, ...)
    # and pmaps across num_devices axis. Assumes batches are consecutive splits
    # of a long time-series.    
    if jax.local_device_count() == 1:
        print(f"WARNING: {jax.local_device_count()} cpu detected.")
    
    dl = SimpleDataloader(emissions,
                          num_devices=jax.local_device_count(),
                          num_batches_per_device=1,
                          num_timesteps_per_batch=1000)

    def em_step(hmm, emissions_iterator):
        normd_suff_stats = streaming_parallel_e_step(hmm, emissions_iterator)
        hmm.m_step(None, normd_suff_stats)
        return normd_suff_stats

    for _ in range(num_iters):
        test_nss = em_step(test_hmm, dl)

    # ----------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    assert jnp.all(get_leading_dim(test_nss)==1)

    assert_allclose_with_errmsg(test_hmm.emission_means,
                                ref_hmm.emission_means,
                                rtol=1e-1, atol=1e-1,
                                title='emission_means')
    
    assert_allclose_with_errmsg(test_hmm.emission_covariance_matrices,
                                ref_hmm.emission_covariance_matrices,
                                rtol=1e-1, atol=1e-1,
                                title='emission_covariance_matrices')

    assert_allclose_with_errmsg(test_nss.marginal_loglik,
                                ref_nss.marginal_loglik,
                                rtol=1e-2, title='marginal_loglik')