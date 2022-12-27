"""Gaussian Hidden Markov Model using normalized gaussian statistics and 
under a normal inverse Wishart (NIW) prior. This codebase is optimized for
performing inference over millions of emissions.

This implementation inherits heavily from the Dynamax codebase, albeit indirectly.
This version removes the object-oriented structure used in the Dynamx codebase
and uses purely functional programming to faciliate improved parallelization
performance during inference. Additional models and generalizations implemented
in the Dynmax codebase are also stripped away for clarity and simplicity.
"""

from ._model import(
    Parameters,
    PriorParameters,
    NormalizedGaussianHMMStatistics,
    initial_distribution,
    transition_distribution,
    emission_distribution,
    log_prior,
    log_prob,
    log_likelihood,
    sample,
)

from ._algorithms import (
    e_step,
    m_step,
    fit_em,
    nonparallel_stochastic_em_step,
    parallel_stochastic_em_step,
    fit_stochastic_em,
    most_likely_states,
)

from ._initialization import(
    initialize_model,
    initialize_prior_from_scalar_values,
    initialize_statistics,
)
