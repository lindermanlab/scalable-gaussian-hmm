""" Legacy code, not actively used or supported.

Includes:
- arg_uiniform_split
- streaming_parallel_e_step
"""

import pytest
import numpy as onp

def arg_uniform_split(target_set_size, set_sizes, set_ids=None):
    """Returns a function that splits the original set of different sizes into
    sets of uniforms sizes, in order. drop_last behavior automatically enforced.

    Suppose we'd like to create even batches of 3 elements from the following
    unevenly-sized sets:
        {a,b,c,d}, {e}, {f,g,h}, {i,j}, {k,l,m,n}
    Each letter is a distinct elemnt within the set. When for-looping through
    this function returns
        [(i=0,s_[0:3]],
        [(i=0,s_[3:4], (i=1,s_[0:1], (i=2,s_[0:1]],
        [(i=2,s_[1:3], (i=3,s_[0:1]],
        [(i=3,s_[1:2], (i=4,s_[0:2]]
    which would result in the sets:
        {a,b,c}, {d,e,f}, {g,h,i}, {j,k,l}

    Example usage:
        num_batches, split_fn = arg_uniform_split(batches_per_file, batch_size)
        i_file = i_in_file = 0      # File index, batch within file index to current file
        for i in range(num_batches):
            b_args, i_file, i_in_file = split_fn([], i_file, i_in_file)

    parameters:
        target_set_size: int
        set_sizes: sequence of ints indicate original set sizes
        set_ids: sequence of ints. Used when shuffling original set sizes
    
    returns:
        list, length (target_num_sets,) of lists with elements (set_id, (frame_start, frame_end))
    """
    def _format(i_set, i_within_start, i_within_end):
        """Defines how resulting args are presented formatted."""
        # return (i_set, np.s_[int(i_within_start):int(i_within_end)])
        return (int(i_set), (int(i_within_start), int(i_within_end)))

    def _fn(n, i_set, i_within_set, out):
        """Recursively returns the args of the sets and elements within sets
        needed to make an even function set.

        Parameters:
            n: int. Number of elements needed to complete an event set.
            i_set: int. Index of current set
            i_within_set: int. Index of current element within set
            out: Current list of arguments

        Returns: 
            i_set: Updated index of current set
            i_within_set: Updated index of unallocated element within set
            out: Updated list of arguments
        """
        n_set = set_sizes[i_set]                   # Size of this set
        
        # This set can complete the remainder of the uniform set: Take as many
        # elements as needed and return to main loop
        if n <= (n_set - i_within_set):
            out.append(_format(set_ids[i_set], i_within_set, i_within_set+n))
            
            # Increase within-set index. If we reached end of set, reset 
            # within-set index to 0 (%) and increase set index
            i_within_set = (i_within_set + n) % n_set
            i_set = i_set + 1 if i_within_set == 0 else i_set
            return i_set, i_within_set, out

        # This set cannot complete the remainder of the uniform set: Take all
        # remaining elements from this set and update number of elements still
        # needed. Then, call function to perform on next set
        else:
            out.append(_format(set_ids[i_set], i_within_set, n_set))
            n -= (n_set - i_within_set)
            return _fn(n, i_set+1, 0, out)

    target_num_sets = sum(set_sizes) // target_set_size                # The number of evenly-sized sets that will result
    set_ids = onp.arange(len(set_sizes)) if set_ids is None else set_ids    
    i_set = i_within_set = 0
    all_args = []
    for _ in range(target_num_sets):
        i_set, i_within_set, set_args = _fn(target_set_size, i_set, i_within_set, [],)
        all_args.append(set_args)
    return all_args

# Testing code -----------------------------------------------------------------
def test_uniform_split():
    """Use elements from multiple sets."""
    # Original sets = [0,1,2,3], [4], [5,6,7], [8,9], [10,11,12,13]
    original_sets = onp.split(onp.arange(14), (4,5,8,10))
    batch_size = 3

    # Get args to split, then split original set
    original_set_sizes = list(map(len, original_sets))
    num_batches = sum(original_set_sizes) // batch_size
    args = arg_uniform_split(batch_size, original_set_sizes)
    uniform_sets = onp.empty((num_batches, batch_size))
    for i_batch, b_args in enumerate(args):
        # uniform_sets = \
            # uniform_sets.at[i_batch] \
            #             .set(onp.concatenate([original_sets[i_set][s_set[0]:s_set[1]] \
            #                                     for (i_set, s_set) in b_args]))
        uniform_sets[i_batch] = onp.concatenate(
            [original_sets[i_set][s_set[0]:s_set[1]] for (i_set, s_set) in b_args]
        )

    expected_sets = onp.arange(12).reshape(4, 3)
    assert onp.all(uniform_sets==expected_sets)

def test_uniform_split_2():
    """Middle sets have clean finishes."""
    # Original sets: [0,1,2,3], [4,5,6,7,8,9], ...
    #                [10,11,12,13,14], [15,16,17,18,19,20,21]
    original_sets = onp.split(onp.arange(22), (4,10,15))
    batch_size = 5

    # Get args to split, then split original set
    original_set_sizes = list(map(len, original_sets))
    num_batches = sum(original_set_sizes) // batch_size
    args = arg_uniform_split(batch_size, original_set_sizes)
    uniform_sets = onp.empty((num_batches, batch_size))
    for i_batch, b_args in enumerate(args):
        # uniform_sets = \
        #     uniform_sets.at[i_batch]\
        #                 .set(onp.concatenate([original_sets[i_set][s_set[0]:s_set[1]] \
        #                                         for (i_set, s_set) in b_args]))
        uniform_sets[i_batch] = onp.concatenate(
            [original_sets[i_set][s_set[0]:s_set[1]] for (i_set, s_set) in b_args]
        )

    expected_sets = onp.arange(20).reshape(4,5)

    assert onp.all(uniform_sets==expected_sets)

# ==============================================================================
import chex
from jax import tree_map, pmap
import jax.numpy as jnp
from functools import partial

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

# ------------------------------------------------------------------------------
    
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
    
    assert jnp.allclose(test_hmm.emission_means.value,
                        ref_hmm.emission_means.value, atol=1)
    assert jnp.allclose(test_hmm.emission_covariance_matrices.value,
                        ref_hmm.emission_covariance_matrices.value, atol=1)
    assert jnp.allclose(test_nss.marginal_loglik,
                        ref_nss.marginal_loglik, atol=100)