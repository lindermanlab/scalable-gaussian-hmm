import chex
import jax.numpy as np

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from ssm_jax.hmm.inference import hmm_smoother

from tensorflow_probability.substrates.jax.distributions import Dirichlet

@chex.dataclass
class NormalizedGaussianHMMSuffStats:
    """Wrapper for normalized sufficient statistics of a GaussianHMM."""
    init_state_probs: chex.Array    # shape (K,)
    transition_probs: chex.Array    # shape (K,K)
    w_normalizer: chex.Array        # shape (K,)
    normd_Ex: chex.Array            # shape (K,D)
    normd_ExxT: chex.Array          # shape (K,D,D)

    @classmethod
    def concat(cls, a, b):
        """Concatenates instances a and b into a new instance of the class."""
        return cls(
            **{k: np.concatenate([a[k], b[k]], axis=0)
               for k in cls.__dataclass_fields__.keys()}
        )

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
    w_normalizer = posterior.smoothed_probs.sum(axis=0)                         # shape (K,). sum_i w_i, shape (K,)
    normd_weights = posterior.smoothed_probs / w_normalizer                                  # shape (T,K)

    normd_Ex = np.einsum('tk, ti->ki', normd_weights, emissions)
    normd_ExxT = np.einsum('tk, ti, tj->kij',
                              normd_weights, emissions, emissions)

    init_state_probs = posterior.smoothed_probs[0]
    transition_probs = posterior.smoothed_transition_probs.sum(axis=0)
    
    return NormalizedGaussianHMMSuffStats(
        init_state_probs=init_state_probs, 
        transition_probs=transition_probs,
        w_normalizer=w_normalizer,
        normd_Ex=normd_Ex,
        normd_ExxT=normd_ExxT
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
    weights = nss.w_normalizer / nss.w_normalizer.sum(axis=0)                                               # shape (M,K,)
    emission_means = (nss.normd_Ex * weights[:,:,None]).sum(axis=0)
    emission_covs  = (nss.normd_ExxT * weights[:,:,None,None]).sum(axis=0) \
                     - np.einsum('ki,kj->kij', emission_means, emission_means) \
                     + 1e-4 * np.eye(emission_dim)

    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_state_prob,
                       transition_matrix,
                       emission_means,
                       emission_covs)

def partial_e_step(hmm: GaussianHMM, emissions: np.ndarray) -> NormalizedGaussianHMMSuffStats:
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
    w_normalizer = posterior.smoothed_probs.sum(axis=0)                         # shape (K,). sum_i w_i, shape (K,)
    normd_weights = posterior.smoothed_probs / w_normalizer                                  # shape (T,K)

    normd_Ex = np.einsum('tk, ti->ki', normd_weights, emissions)
    normd_ExxT = np.einsum('tk, ti, tj->kij',
                              normd_weights, emissions, emissions)

    initial_probs = posterior.smoothed_probs[0]
    transition_probs = posterior.smoothed_transition_probs.sum(axis=0)

    return initial_probs, transition_probs, w_normalizer, normd_Ex, normd_ExxT

def m_step(initial_state_probs, transition_counts, ws, weighted_Ex, weighted_ExxT):
    """Compute ML parameters from batched expected sufficient statistics.

    Inputs:
        initial_state_probs: shape (M,K)
        transition_counts: shape (M,K,K)
        ws: shape (M,K)
        w_Exs: shape (M,K,D)
        w_ExxTs: shape (M,K,D,D)
    Returns:
        hmm: GaussianHMM
    """

    # Initial distribution
    initial_state_prob = Dirichlet(1.0001 + initial_state_probs[0]).mode()

    # Transition distribution
    transition_matrix = Dirichlet(1.0001 + transition_counts.sum(0)).mode()

    # Gaussian emission distribution
    emission_dim = weighted_Ex.shape[-1]
    weights = ws / ws.sum(axis=0)                                               # shape (M,K,)
    emission_means = (weighted_Ex * weights[:,:,None]).sum(axis=0)
    emission_covs  = (weighted_ExxT * weights[:,:,None,None]).sum(axis=0) \
                     - np.einsum('ki,kj->kij', emission_means, emission_means) \
                     + 1e-4 * np.eye(emission_dim)

    # Pack the results into a new GaussianHMM
    return GaussianHMM(initial_state_prob,
                       transition_matrix,
                       emission_means,
                       emission_covs)