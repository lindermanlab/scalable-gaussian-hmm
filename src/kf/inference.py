"""Streaming EM step
E-step computation is parallelized across multiple processors, then
normalized sufficient statistics are combined in M-step
Actual `em_step` which differ based on how emissions are split and thus how
normalized sufficient stats should be combined so that all elements have
leading batch shape (n_splits, n_hmm_states, ...). Here, no recombination
is required.
"""
import chex
import jax
import jax.numpy as jnp

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from tensorflow_probability.substrates.jax.distributions import Dirichlet

# ==============================================================================

@chex.dataclass
class _OnlineSuffStats:
    """Dataclass for cumulatively storing normalized GaussinHMM sufficient statistics."""
    marginal_loglik: chex.Scalar    # shape ([1],)
    initial_probs: chex.Array       # shape ([1],K,)
    trans_probs: chex.Array         # shape ([1],K,K)
    sum_w: chex.Array               # shape ([B],K,)
    normd_x: chex.Array             # shape ([B],K,D,)
    normd_xxT: chex.Array           # shape ([B],K,D,D)

    def __post_init__(self):
        """Verify that all parameters have consistent shapes"""
        assert self.trans_probs.shape[-2:] == (self.num_states, self.num_states)
        assert self.normd_xxT.shape[-2:] == (self.num_obs, self.num_obs)
        assert self.num_states == self.normd_x.shape[-2] == self.normd_xxT.shape[-3] == self.sum_w.shape[-1]
        
        if self.batch_shape:
            assert self.batch_shape == self.normd_x.shape[:-2] == self.normd_xxT.shape[:-3]
        
    @property
    def num_states(self,) -> int:
        return int(self.initial_probs.shape[-1])
    
    @property
    def num_obs(self,) -> int:
        return int(self.normd_x.shape[-1])

    @property
    def batch_shape(self,):
        return self.sum_w.shape[:-1] if self.sum_w.ndim > 1 else ()

    @classmethod
    def empty(cls, shape: tuple):
        """Allocate empty dataclass with specified shapes. Pre-allocated memory for arrays.
    
        `shape` is interpreted as ([B_1, B_2, ...], num_states, num_obs)
        """
        assert len(shape) >= 2, f'Expected tuple of length 2 or more, received {shape}'
        B = shape[:-2]
        K, D = shape[-2:]

        return cls(
            marginal_loglik=jnp.zeros(B),
            initial_probs=jnp.zeros(B+(K,)),
            trans_probs=jnp.zeros(B+(K,K)),
            sum_w=jnp.zeros(B+(K,)),
            normd_x=jnp.empty(B+(K,D,)),
            normd_xxT=jnp.empty(B+(K,D,D)),
        )

    @classmethod
    def convert(cls, other):
        """Convert `other` into a CumulativeSuffStat class instance."""
        return CumulativeSuffStats(
                marginal_loglik=other.marginal_loglik,
                initial_probs=other.initial_probs,
                trans_probs=other.trans_probs,
                sum_w=other.sum_w,
                normd_x=other.normd_x,
                normd_xxT=other.normd_xxT,)

    @classmethod
    def convert_flat(cls, other):
        """Convert `other` into a (flattened) class instance.

        If `other` has batch_dimensions, flattens by summing across batches
        (for HMM parameters) and renormalizing across batches for GMM parameters
        """
        # Else, `other` is a NormalizedGaussianSuffStat. Flatten before return instance
        if other.sum_w.ndim > 1:
            num_states = other.initial_probs.shape[-1]
            num_obs = other.normd_x.shape[-1]

            total_sum_w = other.sum_w.reshape(-1, num_states).sum(axis=0)
            new_normd_w = other.sum_w/total_sum_w
            normd_x = other.normd_x * new_normd_w[...,None]
            normd_x = normd_x.reshape(-1, num_states, num_obs).sum(axis=0)

            normd_xxT = other.normd_xxT * new_normd_w[...,None,None]
            normd_xxT = normd_xxT.reshape(-1, num_states, num_obs, num_obs).sum(axis=0)

            return cls(
                marginal_loglik=other.marginal_loglik.sum(),
                initial_probs=cls.flatten_initial_probs(other.initial_probs),
                trans_probs=other.trans_probs.reshape(-1, num_states, num_states).sum(axis=0),
                sum_w=total_sum_w,
                normd_x=normd_x,
                normd_xxT=normd_xxT,
            )
        else:
            return cls.convert(other)

    def add(self, other):
        # Accumulate the sum
        self.marginal_loglik += other.marginal_loglik
        self.initial_probs += self.accumulate_initial_probs(other.initial_probs)
        self.trans_probs += other.trans_probs

        # Compute running weighted average
        new_sum_w = self.sum_w + other.sum_w
        self.normd_x = (self.sum_w/new_sum_w)[...,None] * self.normd_x \
                       + (other.sum_w/new_sum_w)[...,None] * other.normd_x
        self.normd_xxT = (self.sum_w/new_sum_w)[...,None,None] * self.normd_xxT \
                         + (other.sum_w/new_sum_w)[...,None,None] * other.normd_xxT

        self.sum_w = new_sum_w
        return
    
    @classmethod
    def flatten_initial_probs(cls, other_initial_probs):
        raise NotImplementedError

    def accumulate_initial_probs(self, other_initial_probs):
        raise NotImplementedError

@chex.dataclass
class IndepBatchOnlineSuffStats(_OnlineSuffStats):
    """Data batches treated as independent batches. So, we sum inferred initial_probs"""
    @classmethod
    def flatten_initial_probs(cls, other_initial_probs):
        num_states = other_initial_probs.shape[-1]
        return other.reshape(-1, num_states).sum(axis=0)

    def accumulate_initial_probs(self, other_initial_probs):
        num_states = other_initial_probs.shape[-1]
        return other_initial_probs.reshape(-1, num_states).sum(axis=0)

@chex.dataclass
class SplitBatchOnlineSuffStats(_OnlineSuffStats):
    """Data batches treated as split sequences. So, we only care about the very
    initial initial_probs, and ignore later initial_probs because they are
    actually mid-sequence."""

    @classmethod
    def flatten_initial_probs(cls, other_initial_probs):
        num_states = other_initial_probs.shape[-1]
        return other_initial_probs.reshape(-1, num_states)[0]

    def accumulate_initial_probs(self, other_initial_probs):
        if jnp.all(self.initial_probs==0.):  # If first ijnput, then save the `other_initial_probs`
            return other_initial_probs
        return jnp.zeros_like(self.initial_probs) # Otherwise, ignore

def streaming_parallel_e_step(hmm, emissions_dl):
    """Performs parallelized E-step on a 'streaming' (sequentially-loaded set of emissions).

    This function is useful when the full set of emissions are too large to be
    all loaded into memory at once. Instead, the emissions are loaded in
    (sequential) batches, and an E-step is performed over each batch.
    The sufficient statistics are accumulated with each batch.

    Parameters:
        hmm: GaussianHMM
        emissions_dl: FishPCDataloader

    Returns:
        suff_stats: dataclass containing posterior normalized sufficient statistics
    """
    num_devices = jax.local_device_count()
    parallel_e_step = jax.pmap(hmm.e_step)   

    # Add leading dimension for m-step to sum over
    suff_stats = SplitBatchOnlineSuffStats.empty((hmm.num_states, hmm.num_obs))
    
    p_emiss_shape = (num_devices, emissions_dl.batch_shape[0]//num_devices, *emissions_dl.batch_shape[1:])

    for batch_emissions in emissions_dl:
        # TODO Shape emissions properly IN the dataloader...
        # With JAX device arrays, "this function may in some cases return a copy
        # rather than a view of the input." Normally this is not an issue because
        # jitting the function optimizes memory usage (but obviously we can't).
        # NOTE Memory profiling does not show this to be an issue...
        batch_suff_stats = parallel_e_step(batch_emissions.reshape(*p_emiss_shape))
        batch_suff_stats = SplitBatchOnlineSuffStats.convert_flat(batch_suff_stats)
        suff_stats.add(batch_suff_stats)

    return suff_stats