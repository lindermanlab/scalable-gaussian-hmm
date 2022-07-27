"""Parallelized, streaming online E step

E-step computation is parallelized across multiple processors, then
normalized sufficient statistics are combined in M-step
Actual `em_step` which differ based on how emissions are split and thus how
normalized sufficient stats should be combined so that all elements have
leading batch shape (n_splits, n_hmm_states, ...). Here, no recombination
is required.
"""
import chex
from jax import tree_map, pmap
import jax.numpy as jnp
from functools import partial

# ==============================================================================
@chex.dataclass
class NormalizedGaussianHMMSuffStats:
    # Normalized sufficient statistics of a GaussianHMM
    marginal_loglik: chex.Scalar    # shape (...,)
    initial_probs: chex.Array       # shape (...,K,)
    trans_probs: chex.Array         # shape (...,K,K)
    sum_w: chex.Array               # shape (...,K,)
    normd_x: chex.Array             # shape (...,K,D,)
    normd_xxT: chex.Array           # shape (...,K,D,D)

    @property
    def shape(self,):
        return tree_map(lambda arr: arr.shape, self)

    @classmethod
    def empty(cls, shape: tuple):
        """Allocate empty dataclass with specified shapes."""
        B = shape[:-2]
        K, D = shape[-2:]

        return cls(
            marginal_loglik=jnp.zeros(B),
            initial_probs=jnp.zeros(B+(K,)),
            trans_probs=jnp.zeros(B+(K,K)),
            sum_w=jnp.zeros(B+(K,)),
            normd_x=jnp.zeros(B+(K,D,)),
            normd_xxT=jnp.zeros(B+(K,D,D)),
        )

    @staticmethod
    def flatten(ss):
        """Reshape suff stats dataclass so that fields have shape (-1, K, ...)."""
        K = ss.initial_probs.shape[-1]      # num_states
        D = ss.normd_x.shape[-1]            # emission_dims

        ss.marginal_loglik = ss.marginal_loglik.reshape(-1)
        ss.initial_probs = ss.initial_probs.reshape(-1, K)
        ss.trans_probs = ss.trans_probs.reshape(-1, K, K)

        ss.sum_w = ss.sum_w.reshape(-1, K)
        ss.normd_x = ss.normd_x.reshape(-1, K, D)
        ss.normd_xxT = ss.normd_xxT.reshape(-1, K, D, D)
        return ss

@chex.dataclass
class StreamingSuffStats(NormalizedGaussianHMMSuffStats):
    """Cumulatively store normalized GaussinHMM sufficient statistics."""
    @staticmethod
    def rescale(ss: NormalizedGaussianHMMSuffStats):
        """Rescale a batch of normalized stats in-place. Fields have shape (K,...).
        
        Mathematically the same as the rescaling code used in hmm.m_step.
        Note the special treatment of initial_probs. Since we assume data is seen
        sequentially, we only want to keep the initial probs of the first batch.
        """
        
        ss = NormalizedGaussianHMMSuffStats.flatten(ss)
        
        # ---------------------------------------------
        ss.marginal_loglik = ss.marginal_loglik.sum()
        ss.initial_probs = ss.initial_probs[0]
        ss.trans_probs = ss.trans_probs.sum(axis=0)

        # ---------------------------------------------
        new_sum_w = ss.sum_w.sum(axis=0)
        normd_w = ss.sum_w / new_sum_w
        
        ss.normd_x = (ss.normd_x * normd_w[...,None]).sum(axis=0)
        ss.normd_xxT = (ss.normd_xxT * normd_w[...,None,None]).sum(axis=0)
        ss.sum_w = new_sum_w
        return ss

    def accumulate(self, other):
        """Add the sufficient stats of `other` into the rolling sufficient stats.

        NB the special treatment of initial_probs: Since we assume data is seen
        sequentially, we only the values from the very first set of suff stats.
        Assumes an empty instance is allocated, with initial_probs set to 0.

        Parameters
            other: NormalizedGaussianHMMSuffStats, rescaled so fields have
                   shape (K,...)
        """
        self.marginal_loglik += other.marginal_loglik

        if jnp.all(self.initial_probs==0.):
            self.initial_probs += other.initial_probs 
        self.trans_probs += other.trans_probs

        # Compute running weighted average
        new_sum_w = self.sum_w + other.sum_w
        self.normd_x = (self.sum_w/new_sum_w)[...,None] * self.normd_x \
                       + (other.sum_w/new_sum_w)[...,None] * other.normd_x
        self.normd_xxT = (self.sum_w/new_sum_w)[...,None,None] * self.normd_xxT \
                         + (other.sum_w/new_sum_w)[...,None,None] * other.normd_xxT

        self.sum_w = new_sum_w
        return
    
def streaming_parallel_e_step(hmm, emissions_dl) -> StreamingSuffStats:
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
    stats = StreamingSuffStats.empty((1, hmm.num_states, hmm.num_obs))
    for batch_emissions in emissions_dl:
        batch_stats = pmap(hmm.e_step)(batch_emissions)
        batch_stats = StreamingSuffStats.rescale(batch_stats)
        stats.accumulate(batch_stats)

    return stats