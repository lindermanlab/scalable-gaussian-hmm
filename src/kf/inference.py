import chex
import jax.numpy as np
from jax import tree_map

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from ssm_jax.hmm.inference import (hmm_smoother as _hmm_smoother,
                                   HMMPosterior as _HMMPosterior,
                                   _compute_sum_transition_probs,)

from functools import reduce

from tensorflow_probability.substrates.jax.distributions import Dirichlet

@chex.dataclass
class HMMPosterior(_HMMPosterior):
    """Adds `smoothed_transition_probs_sum` to field.
    
    This field was removed from the dataclass 5/27/22, commit 29b59362e3b5454d91dde32862f76511bf600b3a.
    """
    smoothed_transition_probs_sum: chex.Array = None    # shape (K,K)

def hmm_smoother(initial_distributions, transition_matrix, log_likelihoods):
    """Returns HMMPosterior with `smoothed_transition_probs_sum` field."""

    _posterior =  \
        _hmm_smoother(initial_distributions, transition_matrix, log_likelihoods)
    
    smoothed_transition_probs_sum = \
                    _compute_sum_transition_probs(transition_matrix, _posterior)

    return HMMPosterior(marginal_loglik=_posterior.marginal_loglik,
                        filtered_probs=_posterior.filtered_probs,
                        predicted_probs=_posterior.predicted_probs,
                        smoothed_probs=_posterior.smoothed_probs,
                        smoothed_transition_probs_sum=smoothed_transition_probs_sum)

# ==============================================================================
# Distributed full-batch EM steps
# E-step computation is parallelized across multiple processors, then
# normalized sufficient statistics are combined in M-step
# Actual `em_step` which differ based on how emissions are split and thus how
# normalized sufficient stats should be combined so that all elements have
# leading batch shape (n_splits, n_hmm_states, ...). Here, no recombination
# is required.
#
#   def em_step(hmm, split_emissions):
#       # Do this b/c GaussianHMM doesn't register as pytree correctly yet
#       _e_step = partial(sharded_e_step, hmm)
#       nss = pmap(_e_step)(split_emissions)
#       hmm = fullbatch_m_step(nss)
#       return hmm, nss
# ------------------------------------------------------------------------------

@chex.dataclass
class NormalizedGaussianHMMSuffStats:
    """Wrapper for normalized sufficient statistics of a GaussianHMM.
    
    marginal_log_likelihood and num_emissions fields added for convenience.
    """
    init_state_probs: chex.Array                    # shape ([M],K,)
    transition_probs: chex.Array                    # shape ([M],K,K)
    w_normalizer: chex.Array                        # shape ([M],K,)
    normd_Ex: chex.Array                            # shape ([M],K,D)
    normd_ExxT: chex.Array                          # shape ([M],K,D,D)
    marginal_loglik: float=-np.inf                  # shape ([M],)
    num_emissions: int=0                            # shape ([M],)
    
    @classmethod
    def empty(cls, shape: tuple):
        """Create an empty NGSS object.

        Useful for efficient memory usage by pre-allocating an NGSS object of a
        fixed size. Use in conjunction with the instance function `set`.
        
        Parameters
            shape: tuple
                If length 2, `shape` interpreted as (K,D).
                Else if length 3, `shape` interpreted as (M,K,D)
        Returns
            ngss: NGSS instance with empty fields of appropriate shapes
        """

        if len(shape) == 2:
            K, D = shape
            return cls(
                init_state_probs=np.empty((K,)),
                transition_probs=np.empty((K,K,)),
                w_normalizer=np.empty((K,)),
                normd_Ex=np.empty((K,D,)),
                normd_ExxT=np.empty((K,D,D,)),
            )
        elif len(shape) == 3:
            M, K, D = shape
            return cls(
                init_state_probs=np.empty((M,K)),
                transition_probs=np.empty((M,K,K)),
                w_normalizer=np.empty((M,K)),
                normd_Ex=np.empty((M,K,D)),
                normd_ExxT=np.empty((M,K,D,D)),
                marginal_loglik=np.empty((M,)),
                num_emissions=np.empty((M,)),
            )
        else:
            raise ValueError(f"Expected tuple of length 2 or 3, received {len(shape)}")

    def batch_set(self, slice, other):
        """Set field values at indicated indices to field vlaues of `other`.
        
        Parameters:
            slice: int or 1D index slice
                "Length" of slice must equal M (or 1 if `other` isn't batched)
            other: NGSS, with shape ([M],...)
        """
        for k in self.__dataclass_fields__.keys():
            setattr(self, k, self[k].at[slice].set(other[k]))
        return 

    def batch_marginal_loglik(self,):
        """Compute average marginal log likelihood from batch of normalized 
        marginal loglikelihoods. Assumes that this is a batched NGSS instance,
        i.e. data fields have dimensions (M,...).

        Should be numerically safe - If number is extremely off, check for overflow.
        """
        ns = self.num_emissions
        return (ns/ns.sum() * self.marginal_loglik).sum()/ns.sum()
    
    @classmethod
    def concat(cls, a, b, axis=0):
        """Concatenates instances a and b into a new instance of the class.
        
        NB: Both `a` and `b` must have leading batch dimension.
        NB: Convenient, but can be memory inefficient if calling many times. See
        instead `cls.empty(shape)` and `self.batch_set` functions
        """
        return cls(
            **{k: np.concatenate([a[k], b[k]], axis=axis)
               for k in cls.__dataclass_fields__.keys()}
        )

    @classmethod
    def stack(cls, ss_seq, axis=0):
        """Stack a sequence of class objects along new axis."""
        _ss_seq = tree_map(lambda arr: np.expand_dims(arr, axis=axis), ss_seq)
        return reduce(cls.concat, _ss_seq)

def sharded_e_step(hmm: GaussianHMM, emissions: chex.Array) -> NormalizedGaussianHMMSuffStats:
    """Computes posterior expected sufficient stats given partial batch of emissions.

    Inputs
        hmm: GaussianHMM with K states
        emissions: shape (T_m, D)
    Returns
        NormalizedGaussianHMMSuffStats dataclass
    """

    # Compute posterior HMM distribution
    lps = hmm.emission_distribution.log_prob(emissions[..., None, :])
    posterior =  \
            hmm_smoother(hmm.initial_probabilities, hmm.transition_matrix, lps)

    # Compute expected sufficient statistics
    # Sometimes, w_normalizer = 0, resulting in NaNs in normd_Ex and normed_ExxT
    w_normalizer = posterior.smoothed_probs.sum(axis=0)                         # shape (K,). sum_i w_i, shape (K,)
    normd_weights = posterior.smoothed_probs / w_normalizer                     # shape (T,K)

    emission_dim = emissions.shape[-1]
    normd_Ex = np.where(w_normalizer[:,None] < 1e-6,
                        np.zeros(emission_dim),
                        np.einsum('tk, ti->ki', normd_weights, emissions))

    normd_ExxT = np.where(w_normalizer[...,None,None] < 1e-6,
                          1e6 * np.eye(emission_dim),
                          np.einsum('tk, ti, tj->kij', normd_weights, emissions, emissions))

    init_state_probs = posterior.smoothed_probs[0]
    transition_probs = posterior.smoothed_transition_probs_sum
    
    return NormalizedGaussianHMMSuffStats(
        init_state_probs=init_state_probs, 
        transition_probs=transition_probs,
        w_normalizer=w_normalizer,
        normd_Ex=normd_Ex,
        normd_ExxT=normd_ExxT,
        marginal_loglik=posterior.marginal_loglik,
        num_emissions=len(emissions)
    )

def collective_m_step(nss: NormalizedGaussianHMMSuffStats) -> GaussianHMM:
    """Compute ML parameters from sharded expected sufficient statistics.

    Inputs:
        nss: NormalizedGaussianHMMSuffStats dataclass, each element with leading
        dimension (M,...)
        
    Returns:
        hmm: GaussianHMM
    """

    # Initial distribution
    initial_state_prob = Dirichlet(1.0001 + nss.init_state_probs[0]).mode()

    # Transition distribution
    transition_matrix = Dirichlet(1.0001 + nss.transition_probs.sum(0)).mode()

    # Gaussian emission distribution
    emission_dim = nss.normd_Ex.shape[-1]
    weights = nss.w_normalizer / nss.w_normalizer.sum(axis=0)                   # shape (M,K,)

    emission_means = (nss.normd_Ex * weights[:,:,None]).sum(axis=0)
    emission_covs  = (nss.normd_ExxT * weights[:,:,None,None]).sum(axis=0) \
                     - np.einsum('ki,kj->kij', emission_means, emission_means) \
                     + 1e-4 * np.eye(emission_dim)

    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_state_prob,
                       transition_matrix,
                       emission_means,
                       emission_covs)

# ==============================================================================
# "Standard" full-batch EM steps
#
#   def em_step(hmm, emissions):
#       posterior = fullbatch_e_step(hmm, emissions)
#       hmm = fullbatch_m_step(posterior, emissions)
#       return hmm, posterior
# ------------------------------------------------------------------------------
def fullbatch_e_step(hmm, emissions):
    return hmm_smoother(hmm.initial_probabilities,
                        hmm.transition_matrix,
                        hmm.emission_distribution.log_prob(emissions[..., None, :]))

def fullbatch_m_step(posterior, emissions):
    # Initial distribution
    initial_probs = Dirichlet(1.0001 + posterior.smoothed_probs[0]).mode()

    # Transition distribution
    transition_matrix = Dirichlet(
        1.0001 + posterior.smoothed_transition_probs_sum).mode()

    # Gaussian emission distribution
    w_sum = np.einsum('tk->k', posterior.smoothed_probs)
    x_sum = np.einsum('tk, ti->ki', posterior.smoothed_probs, emissions)
    xxT_sum = np.einsum('tk, ti, tj->kij', posterior.smoothed_probs, emissions, emissions)

    emission_means = x_sum / w_sum[:, None]
    emission_covs = xxT_sum / w_sum[:, None, None] \
        - np.einsum('ki,kj->kij', emission_means, emission_means) \
        + 1e-4 * np.eye(emissions.shape[1])
    
    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_probs,
                       transition_matrix,
                       emission_means,
                       emission_covs)