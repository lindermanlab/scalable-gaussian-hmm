"""Parallelized, streaming online E step

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
from functools import partial

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from tensorflow_probability.substrates.jax.distributions import Dirichlet

# ==============================================================================

@chex.dataclass
class StreamingSuffStats:
    """Dataclass for cumulatively storing normalized GaussinHMM sufficient statistics."""
    marginal_loglik: chex.Scalar    # shape ([1],)
    initial_probs: chex.Array       # shape ([1],K,)
    trans_probs: chex.Array         # shape ([1],K,K)
    sum_w: chex.Array               # shape ([B],K,)
    normd_x: chex.Array             # shape ([B],K,D,)
    normd_xxT: chex.Array           # shape ([B],K,D,D)

    # def __post_init__(self):
    #     """Verify that all parameters have consistent shapes"""
    #     assert self.trans_probs.shape[-2:] == (self.num_states, self.num_states)
    #     assert self.normd_xxT.shape[-2:] == (self.num_obs, self.num_obs)
    #     assert self.num_states == self.normd_x.shape[-2] == self.normd_xxT.shape[-3] == self.sum_w.shape[-1]
        
    #     # if self.batch_shape:
        #     assert self.batch_shape == self.normd_x.shape[:-2] == self.normd_xxT.shape[:-3]
        
    @property
    def num_states(self,) -> int:
        return int(self.initial_probs.shape[-1])
    
    @property
    def num_obs(self,) -> int:
        return int(self.normd_x.shape[-1])

    @property
    def shape(self,):
        return jax.tree_map(lambda arr: arr.shape, self)

    # @property
    # def batch_shape(self,):
    #     return self.sum_w.shape[:-1] if self.sum_w.ndim > 1 else ()

    @classmethod
    def empty(cls, shape: tuple):
        """Allocate empty dataclass with specified shapes. Pre-allocated memory for arrays.
    
        `shape` is interpreted as ([B_1, B_2, ...], num_states, num_obs)
        """
        # assert len(shape) >= 2, f'Expected tuple of length 2 or more, received {shape}'
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
    
    def flatten(self,):
        """Reshapes all fields to have 1D batch
        
        Resulting fields have shape (B,K,...).
        """
        self.marginal_loglik = self.marginal_loglik.reshape(-1)
        self.initial_probs = self.initial_probs.reshape(-1, self.num_states)
        self.trans_probs = self.trans_probs.reshape(-1, self.num_states, self.num_states)

        self.sum_w = self.sum_w.reshape(-1, self.num_states)
        self.normd_x = self.normd_x.reshape(-1, self.num_states, self.num_obs)
        self.normd_xxT = self.normd_xxT.reshape(-1, self.num_states, self.num_obs, self.num_obs)
        return

    def normalize(self, ):
        """Normalizes fields across the batch.
        
        Resulting fields have shape (K,...).
        
        NB special treatment of initial_probs: Since we assume that data is
        streamed sequentially, initial_probs should be entirely determined
        by the initial_probs of the first batch.
        """
        self.flatten()

        self.marginal_loglik = self.marginal_loglik.sum()
        self.initial_probs = self.initial_probs[0]
        self.trans_probs = self.trans_probs.sum(axis=0)

        new_sum_w = self.sum_w.sum(axis=0)
        normd_w = self.sum_w / new_sum_w
        
        self.normd_x = (self.normd_x * normd_w[...,None]).sum(axis=0)
        self.normd_xxT = (self.normd_xxT * normd_w[...,None,None]).sum(axis=0)
        self.sum_w = new_sum_w
        return

    def add(self, other):
        # Accumulate the sum
        self.initial_probs += other.initial_probs if jnp.all(self.initial_probs==0.) \
                              else jnp.zeros_like(other.initial_probs)
        self.marginal_loglik += other.marginal_loglik
        self.trans_probs += other.trans_probs

        # Compute running weighted average
        new_sum_w = self.sum_w + other.sum_w
        self.normd_x = (self.sum_w/new_sum_w)[...,None] * self.normd_x \
                       + (other.sum_w/new_sum_w)[...,None] * other.normd_x
        self.normd_xxT = (self.sum_w/new_sum_w)[...,None,None] * self.normd_xxT \
                         + (other.sum_w/new_sum_w)[...,None,None] * other.normd_xxT

        self.sum_w = new_sum_w
        return
    
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
    stats = StreamingSuffStats.empty((1, hmm.num_states, hmm.num_obs))

    for batch_emissions in emissions_dl:
        batch_stats = jax.pmap(hmm.e_step)(batch_emissions)
        rescaled_stats = StreamingSuffStats(
            marginal_loglik=batch_stats.marginal_loglik,
            initial_probs=batch_stats.initial_probs,
            trans_probs=batch_stats.trans_probs,
            sum_w=batch_stats.sum_w,
            normd_x=batch_stats.normd_x,
            normd_xxT=batch_stats.normd_xxT,
        )

        rescaled_stats.normalize()

        stats.add(rescaled_stats)
    # from itertools import islice
    # import pdb
    # for batch_emissions in emissions_dl:
    #     batch_stats = sharded_e_step(hmm, batch_emissions)
    #     batch_stats.normalize()
    #     stats.add(batch_stats)

    # # all close EXCEPT
    # stats2.sum_w == stats.sum_w / 2
    # stats2.trans_prob == stats.trans_probs / 2
    # stats2 = StreamingSuffStats.empty((1, hmm.num_states, hmm.num_obs))
    # for batch_emissions in emissions_dl:
    #     batch_stats2 = sharded_e_step2(hmm, batch_emissions)
    #     batch_stats2.normalize()
    #     stats2.add(batch_stats2)

    # import pdb; pdb.set_trace()


    return stats