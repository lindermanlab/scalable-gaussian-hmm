import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
from jax.tree_util import tree_map, tree_flatten, tree_leaves
from functools import partial
import optax
from tqdm.auto import trange, tqdm

from dynamax.hidden_markov_model.inference import (
    hmm_smoother, hmm_posterior_mode, compute_transition_probs
)
import snax.checkpoint as chk

from kf.gaussian_hmm._initialization import initialize_statistics
from kf.gaussian_hmm._model import *

__all__ = [
    'e_step',
    'm_step',
    'fit_em',
    'fit_stochastic_em',
    'most_likely_states',
]

# ==============================================================================
# EXPECTATION FUNCTIONS
# ==============================================================================

def e_step(params, batched_emissions):
    """Compute normalized expected sufficient statistics under the posterior.    

    Arguments
        params (Parameters)
        batched_emissions[b,t,d]

    Returns
        reduced_normd_stats (NormalizedGaussianHMMStatistics([k,...]))
        reduced_normalizer (ndarray[k,])
        reduced_marginal_loglik (float)
    """

    def _single_e_step(emissions):
        # Run the smoother to calculate the posterior
        posterior = hmm_smoother(params.initial_probs,
                                 params.transition_probs,
                                 log_likelihood(params, emissions))

        # Compute summed Gaussian HMM statististics
        weights = posterior.smoothed_probs
        summed_stats = NormalizedGaussianHMMStatistics(
            initial_pseudocounts = posterior.initial_probs,
            transition_pseudocounts = compute_transition_probs(params.transition_probs, posterior),
            emission_weights=jnp.sum(weights, axis=0),
            emission_xxT=jnp.einsum("tk,ti,tj->kij", weights, emissions, emissions),
            emission_x=jnp.einsum("tk,ti->ki", weights, emissions),
        )
        
        # Normalize statistics by len(emissions). Perform after summation since
        # we typically work in the regime of (len(emissions) >> 1 > posterior probs)
        normd_stats = tree_map(lambda stat: stat / len(emissions), summed_stats)

        num_states = weights.shape[-1]
        normalizer = len(emissions) * jnp.ones(num_states)

        return normd_stats, normalizer, posterior.marginal_loglik

    # Map the E-step calculations over the batch dimension
    batched_normd_stats, batched_normalizer, batched_marginal_logliks \
                                        = vmap(_single_e_step)(batched_emissions)

    # Reduce batched outputs along batch dimension. Since normalizer is constant
    # across all batches, then sufficient statistics are simply an average
    # across batch dimensions, and normalizer is summed to account for this.
    reduced_normd_stats = tree_map(partial(jnp.mean, axis=0), batched_normd_stats)
    reduced_normalizer = jnp.sum(batched_normalizer, axis=0)
    reduced_marginal_logliks = jnp.sum(batched_marginal_logliks)

    return reduced_normd_stats, reduced_normalizer, reduced_marginal_logliks

# ==============================================================================
# MAXIMIZATION FUNCTIONS
# ==============================================================================

def niw_convert_mean_to_natural(loc, conc, df, scale):
    """Convert NIW mean parameters to natural parameters."""
    dim = loc.shape[-1]
    eta_1 = df + dim + 2
    eta_2 = scale + conc * jnp.outer(loc, loc)
    eta_3 = conc * loc
    eta_4 = conc
    return eta_1, eta_2, eta_3, eta_4

def niw_convert_natural_to_mean(eta_1, eta_2, eta_3, eta_4):
    """Convert NIW natural parameters to mean parameters."""
    dim = eta_3.shape[-1]
    loc = eta_3 / eta_4
    conc = eta_4            
    scale = eta_2 - jnp.outer(eta_3, eta_3) / eta_4
    df = eta_1 - dim - 2
    return loc, conc, df, scale

def m_step(prior_params, normalized_stats, normalizer):
    """Compute MAP estimate of Gaussian HMM parameters.

    Implicitly assumes that num_states > 1.

    Arguments
        prior_params (PriorParameters): Parameter values of prior distributions
        normalized_stats (NormalizedGaussianHMMStatistics([k,...]))
        normalizer (ndarray([k,]))
    
    Returns
        map_params (Parameters)
    """

    normd_one = jnp.nan_to_num(1./normalizer, nan=0.0)
    dirichlet_mode = (lambda normd_alpha: 
        (normd_alpha - normd_one) / jnp.sum(normd_alpha - normd_one, axis=-1, keepdims=True)
    )

    # Calculate mode of posterior initial distribution (Dirichlet)
    posterior_initial_conc = (
        normalized_stats.initial_pseudocounts
        + jnp.nan_to_num(prior_params.initial_probs_conc / normalizer, nan=0.0)
    )
    initial_probs = dirichlet_mode(posterior_initial_conc)

    # Calculate mode of posterior transition distribution (Dirichlet)
    posterior_transition_conc = (
        normalized_stats.transition_pseudocounts \
        + jnp.nan_to_num(prior_params.transition_probs_conc / normalizer, nan=0.0)
    )
    transition_probs = dirichlet_mode(posterior_transition_conc)

    # Calculate mode of posterior emission distribution (NIW)
    def _single_emission_m_step(prior_params, normd_stats, norm):
        # Convert prior NIW parameters to natural parameterization
        # TODO Store natural parameterization in prior params (see note below)
        natural_prior_params = niw_convert_mean_to_natural(*prior_params)

        # Normalize prior parameters
        normd_natural_prior_params \
            = tree_map(lambda eta: jnp.nan_to_num(eta / norm, nan=0.0), natural_prior_params)

        # Compute posterior parameters
        normd_emission_suff_stats = (
            normd_stats.emission_weights, normd_stats.emission_xxT,
            normd_stats.emission_x, normd_stats.emission_weights)
        normd_natural_posterior_params \
            = tree_map(jnp.add, normd_natural_prior_params, normd_emission_suff_stats)
        
        # Convert natural posterior parameters to mean parameterization
        posterior_loc, _, _, posterior_scale \
            = niw_convert_natural_to_mean(*normd_natural_posterior_params)

        # Return modal values of posterior distribution
        modal_cov = posterior_scale / normd_natural_posterior_params[0]
        modal_mean = posterior_loc

        return modal_cov, modal_mean

    # Map the emissions M-step calculation over the state dimension
    # TODO Store natural parameterization in prior params (see note above)
    prior_niw_mean_params = (
        prior_params.emission_loc, prior_params.emission_conc,
        prior_params.emission_df, prior_params.emission_scale
    )
    covs, means = vmap(_single_emission_m_step)(
        prior_niw_mean_params, normalized_stats, normalizer
    )

    return Parameters(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_means=means,
        emission_covariances=covs,
    )
    
# ==============================================================================
#
# FULL-BATCH EM ALGORITHM
#
# ==============================================================================

def fit_em(initial_params, prior_params, batched_emissions, num_epochs=50, verbose=True):
    """Estimate model parameters from emissions using Expectation-Maximization (EM).
    
    Arguments
        initial_params (Parameters([k,...]))
        prior_params (PriorParams([k,...]))
        batch_emissions ([b,t,d])
        num_epochs (int): Number of EM iterations to run over full dataset
        verbose (bool): If true, print progress bar.
    
    Returns
        fitted_params (Parameters([k,...]))
        lps (ndarray[num_epochs,])
    """

    @jit
    def em_step(params):
        # Compute expected sufficient statistics
        normalized_stats, normalizer, lls = e_step(params, batched_emissions)

        # Compute MAP estimate
        map_params = m_step(prior_params, normalized_stats, normalizer)
        
        # calculate log likelihood
        lp = log_prior(params, prior_params) + lls

        return map_params, lp

    log_probs = []
    params = initial_params
    pbar = trange(num_epochs) if verbose else range(num_epochs)
    for _ in pbar:
        params, marginal_loglik = em_step(params)
        log_probs.append(marginal_loglik)
    return params, jnp.array(log_probs)

# ==============================================================================
#
# STOCHASTIC EM ALGORITHM
#
# ==============================================================================

@jit
def nonparallel_stochastic_em_step(prior_params, params, rolling_stats,
                                   learning_rate, minibatch_emissions):

    """Perform single stochastic EM step using a single core.
    
    Arguments:
        prior_params (PriorParameters)
        params (Parameters)
        rolling_stats (NormalizedGaussianHMMStatistics)
        learning_rate (float)
        minibatch_emissions (jnp.ndarray): shape [m,t,d]
    
    Returns:
        updated_params (Parameters)
        updated_rolling_stats (NormalizedGaussianHMMStatistics)
        minibatch_lls (float): Expected log-likelihood of minibatch
    """
    # Returns: Parameters([k,...]), *Stats([k,...]), float

    rolling_normalized_stats, rolling_normalizer = rolling_stats

    # Compute the sufficient stats given a minibatch of emissions
    minibatch_normalized_stats, minibatch_normalizer, minibatch_lls \
                                    = e_step(params, minibatch_emissions)

    # Convexly combine minibatch statistics into rolling statistics
    _convex_update = lambda rolling_stat, minibatch_stat: \
        (1-learning_rate) * rolling_stat + learning_rate * minibatch_stat
    
    updated_normalized_stats = tree_map(
        _convex_update, rolling_normalized_stats, minibatch_normalized_stats)

    updated_normalizer = _convex_update(rolling_normalizer, minibatch_normalizer)

    updated_rolling_stats = (updated_normalized_stats, updated_normalizer)

    # Call M-step
    updated_params = m_step(prior_params, updated_normalized_stats, updated_normalizer)

    return updated_params, updated_rolling_stats, minibatch_lls

def parallel_stochastic_em_step(prior_params, params, rolling_stats,
                                learning_rate, minibatch_emissions):
    """Perform single step of stochastic EM using multiple cores.
    
    Inputs and outputs are the same as nonparallel_stochastic_em_step,
    except that minibatch_emissions has shape [p,m,t,d].
    """


    @partial(pmap, in_axes=(None, None, None, None, 0), axis_name='p')
    def _parallel_stochastic_em_step(prior_params, params, rolling_stats,
                                     learning_rate, local_minibatch_emissions):
        
        rolling_normalized_stats, rolling_normalizer = rolling_stats

        # INPUT: local_minibatch_emissions[m,t,d] varying across outer axis 'p'
        # OUTPUT: NamedTuples with leading axis [k,...], varying across outer axis 'p'
        local_normalized_stats, local_normalizer, local_lls \
                                        = e_step(params, local_minibatch_emissions)
        
        # Reduce batch statistics across outer batch axis 'p'
        # OUTPUT: NamedTuples have leading axis [k,...], identical across outer axis 'p'
        collective_normalized_stats = tree_map(
            partial(lax.pmean, axis_name='p'), local_normalized_stats)

        collective_normalizer = lax.psum(local_normalizer, axis_name='p')
        collective_lls = lax.psum(local_lls, axis_name='p')

        # Convexly combine rolling statistics with collective statistics
        # All stats are identical across outer axis 'p'
        convex_update = lambda rolling_stat, minibatch_stat: \
            (1-learning_rate) * rolling_stat + learning_rate * minibatch_stat
        
        p_updated_normalized_stats = tree_map(
            convex_update, rolling_normalized_stats, collective_normalized_stats)

        p_updated_normalizer = convex_update(rolling_normalizer, collective_normalizer)

        p_updated_rolling_stats = (p_updated_normalized_stats, p_updated_normalizer)
        
        # Call M-step
        p_updated_params = m_step(prior_params, p_updated_normalized_stats, p_updated_normalizer)

        return p_updated_params, p_updated_rolling_stats, collective_lls

    pbatch_params, pbatch_stats, pbatch_lls = _parallel_stochastic_em_step(
            prior_params, params, rolling_stats, learning_rate, minibatch_emissions)

    updated_params, updated_rolling_stats, minibatch_lls = tree_map(
        lambda arr: arr[0], (pbatch_params, pbatch_stats, pbatch_lls))

    return updated_params, updated_rolling_stats, minibatch_lls

def fit_stochastic_em(initial_params, prior_params, emissions_generator,
                      schedule=None, num_epochs=5, parallelize=False,
                      checkpoint_every=None, checkpoint_dir=None, num_checkpoints_to_keep=-1):
    """Estimate model parameters from emissions using stochastic Expectation-Maximization (StEM).

    Let the original dataset consists of N independent sequences of length T.
    The StEM algorithm then performs EM on each random subset of M sequences
    (not timesteps) during each epoch. Specifically, it will perform N//M
    iterations of EM per epoch. The algorithm uses a learning rate schedule
    to anneal the minibatch sufficient statistics at each stage of training.
    If a schedule is not specified, an exponentially decaying model is used
    such that the learning rate which decreases by 5% at each epoch.

    NB: This algorithm assumes that the `emissions_generator` object automatically
    shuffles minibatch sequences before each epoch. It is up to the user to
    correctly instantiate this object to exhibit this property. For example,
    `torch.utils.data.DataLoader` objects implement such a functionality.

    If `parallelize=True`, then this algorithm makes use of JAX's pmap and
    collective all-reduce operationsto perform the expensive E-steps in parallel.
    It assumes that there are 'p' devices, which must EXACTLY match the 'p' in
    the minibatch shape returned by the emissions_generator iterable.

    Currently, this code assumes parallelization over multiple CPU cores on the
    same devices. This requires the user to set the environment flags
        XLA_FLAGS=--xla_force_host_platform_device_count=[p]
    
    Arguments
        initial_params (Parameters)
        prior_params (PriorParams)
        emissions_generator (Generator->[m,t,d] OR -> [p,m,t,d] if parallelize):
            Produces minibatches of emissions. Automatically shuffles after each epoch.
        schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
            schedule; defaults to exponential schedule.
        num_epochs (int): Number of StEM iterations to run over full dataset
        parallelize (bool): If True, parallelize across p devices.
        checkpoint_every (int): Number of epochs between checkpoints. Must be
            specified to enable checkpointing.
        checkpoint_dir (str): Directory to story checkpoints. Must be specified
            to enable checkpointing.
        checkpoints_to_keep (int): Number of recent checkpoints to keep.
            Default: -1, keep all checkpoints.
    
    Returns
        fitted_params (Parameters)
        log_probs [num_epochs, num_batches]
    """
    
    num_batches = len(emissions_generator)
    num_states = initial_params.initial_probs.shape[-1]
    emission_dim = initial_params.emission_means.shape[-1]

    # =========================================================================
    # Set global training learning rates: shape (num_epochs, num_batches)
    # =========================================================================
    if schedule is None:
        schedule = optax.exponential_decay(
            init_value=1., 
            end_value=0.,
            transition_steps=num_batches,
            decay_rate=.95,
        )

    learning_rates = schedule(jnp.arange(num_epochs * num_batches))
    assert learning_rates[0] == 1.0, "Learning rate must start at 1."
    learning_rates = learning_rates.reshape(num_epochs, num_batches)

    # =========================================================================
    # Define which method (parallel vs. non-parallel) to use
    # =========================================================================
    step_fn = parallel_stochastic_em_step if parallelize else nonparallel_stochastic_em_step

    # =========================================================================
    # Initialize
    # =========================================================================
    params = initial_params
    rolling_stats = initialize_statistics(num_states, emission_dim)
    expected_log_probs = jnp.empty((0, len(emissions_generator)))
    
    # Load model from latest checkpoint, if checkpointing
    global_start_epoch = 0
    do_checkpoint = (checkpoint_every is not None) and (checkpoint_dir is not None)
    if do_checkpoint:
        _, treedef = tree_flatten((params, rolling_stats, expected_log_probs))
        out = chk.load_latest_checkpoint_with_treedef(treedef, checkpoint_dir)
        if out[0] is not None:
            (params, rolling_stats, expected_log_probs), global_start_epoch = out
            print(f"Loaded checkpoint at step {global_start_epoch} from {checkpoint_dir}.")
        else:
            print("Cold-starting from passed-in parameters.")

    # =========================================================================
    # Train
    # =========================================================================
    epoch = global_start_epoch
    while epoch < num_epochs:
        epoch_expected_lps = []

        pbar = tqdm(emissions_generator,
                    desc=f'epoch {epoch}/{num_epochs}',
                    postfix={'lp': -jnp.inf})

        for minibatch, minibatch_emissions in enumerate(pbar):
            params, rolling_stats, minibatch_lls, _debug_vals = step_fn(
                prior_params, params, rolling_stats, learning_rates[epoch][minibatch],
                minibatch_emissions
            )

            # Store expected log probability, averaged across total number of emissions
            expected_lp = log_prior(params, prior_params) + num_batches * minibatch_lls
            expected_lp /= emissions_generator.total_emissions
            epoch_expected_lps.append(expected_lp)

            # Updated progress bar
            pbar.set_postfix({'lp': expected_lp})

            # Save results, if checkpointing
            # TODO Fix discrepancy between inner (minibatch) and outer (epoch) step
            if do_checkpoint and (epoch % checkpoint_every == 0):
                tqdm.write(f"Saving checkpoint for epoch {epoch}, minibatch {minibatch} at {checkpoint_dir}, number {epoch*num_batches+minibatch}...", end="")
                chk.save_checkpoint((params, rolling_stats, minibatch_lls,),
                    epoch*num_batches+minibatch, checkpoint_dir, num_checkpoints_to_keep)
                tqdm.write("Done.")

            # Check that M-step emission covariance is PSD
            _eigvals = jnp.linalg.eigvals(params.emission_covariances)
            if jnp.any(_eigvals < 0.):
                print('!! `params.emission_covariances` is not PSD. !!')
                for _i, _v in zip(jnp.atleast_2d(jnp.hstack(jnp.nonzero(_eigvals<0))),
                                  _eigvals[_eigvals<0]):
                    print(f'Eigenvalue {_v:.2e} at index {_i}')
                raise ValueError('`params.emission_covariances` is not PSD. Considering increasing emission_scale and emission_extra_df prior parameters')

            # Check no NaNs in parameters
            if any(tree_leaves(tree_map(lambda arr: jnp.any(jnp.isnan(arr)), (params, rolling_stats, minibatch_lls, _debug_vals)))):
                raise ValueError(f'Epoch {epoch}, minibatch {minibatch}: NaN detected in parameters.')

        # Save epoch mean of expected log probs
        expected_log_probs = jnp.vstack([expected_log_probs, jnp.asarray(epoch_expected_lps)])

        epoch += 1

    return params, jnp.asarray(expected_log_probs)

# ==============================================================================
#
# VITERBI PATH
#
# ==============================================================================

def most_likely_states(params, emissions):
    """Compute Viterbi path for observed emissions."""

    return  hmm_posterior_mode(params.initial_probs,
                               params.transition_probs,
                               log_likelihood(params, emissions))