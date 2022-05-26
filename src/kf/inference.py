import jax.numpy as np

from ssm_jax.hmm.models import GaussianHMM                                      # probml/ssm-jax : https://github.com/probml/ssm-jax
from ssm_jax.hmm.inference import hmm_smoother, HMMPosterior

from tensorflow_probability.substrates.jax.distributions import Dirichlet

def partial_e_step(hmm: GaussianHMM, emissions: np.ndarray):
    """Computes posterior expected sufficient stats given partial emissions.

    Inputs
        hmm: GaussianHMM with K states
        emissions: shape (T_m, D)

    Returns
        initial_prob: shape (K,)
        transition_counts: shape (K,K)
        weights_sum: shape (K,)
        weighted_avg_x: shape (K,D)
            Weighted 1st sufficient statistic, averaged over emissions
        weighted_avg_xxT: shape (K,D,D)
            Weighted 2nd sufficient statistic, averaged over emissions
    """

    # Compute posterior HMM distribution
    lps = hmm.emission_distribution.log_prob(emissions[..., None, :])
    posterior =  \
            hmm_smoother(hmm.initial_probabilities, hmm.transition_matrix, lps)

    # Compute expected sufficient statistics
    w_sum = posterior.smoothed_probs.sum(axis=0)                                # shape (K,)                                           # sum_i w_i, shape (K,)
    weights = posterior.smoothed_probs / w_sum                                  # shape (T,K)

    weighted_Ex = np.einsum('tk, ti->ki', weights, emissions)
    weighted_ExxT = np.einsum('tk, ti, tj->kij',
                              weights, emissions, emissions)

    initial_prob = posterior.smoothed_probs[0]
    transition_counts = posterior.smoothed_transition_probs.sum(axis=0)

    return initial_prob, transition_counts, w_sum, weighted_Ex, weighted_ExxT

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