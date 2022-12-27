"""Fit HMM to killifish data using stochastic EM algorithm.

The fit_stochastic_em function here overrides the fitting algorithm in the
gaussian_hmm library because it incorporates dataset-specific Dataloader information.
This addition allows us to save intermediate checkpoints and warm-start the
training from intra-epoch iterations, which is highly desirable for this large
dataset.
"""

import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
from jax.tree_util import tree_map, tree_flatten, tree_leaves
from functools import partial
import optax
from tqdm.auto import trange, tqdm

import snax.checkpoint as chk

import kf.gaussian_hmm as gaussian_hmm

def check_psd(covs):
    """Raise ValueError if covariance matrices are not positive semi-definite."""
    _eigvals = jnp.linalg.eigvals(covs)
    if jnp.any(_eigvals < 0.):
        print('!! `params.emission_covariances` is not PSD. !!')
        for _i, _v in zip(jnp.atleast_2d(jnp.hstack(jnp.nonzero(_eigvals < 0))),
                            _eigvals[_eigvals<0]):
            print(f'Eigenvalue {_v:.2e} at index {_i}')
        raise ValueError('`params.emission_covariances` is not PSD. Considering increasing emission_scale and emission_extra_df prior parameters')

def fit_stochastic_em(initial_params, prior_params,
                      emissions_loader,
                      schedule=None, num_epochs=5, parallelize=False,
                      checkpoint_every=None, checkpoint_dir=None, num_checkpoints_to_keep=10):
    """Estimate model parameters from emissions using stochastic Expectation-Maximization (StEM).

    Let the original dataset consists of N independent sequences of length T.
    The StEM algorithm then performs EM on each random subset of M sequences
    (not timesteps) during each epoch. Specifically, it will perform N//M
    iterations of EM per epoch. The algorithm uses a learning rate schedule
    to anneal the minibatch sufficient statistics at each stage of training.
    If a schedule is not specified, an exponentially decaying model is used
    such that the learning rate which decreases by 5% at each epoch.

    NB: This algorithm assumes that the `emissions_loader` object automatically
    shuffles minibatch sequences before each epoch. It is up to the user to
    correctly instantiate this object to exhibit this property. For example,
    `torch.utils.data.DataLoader` objects implement such a functionality.

    If `parallelize=True`, then this algorithm makes use of JAX's pmap and
    collective all-reduce operationsto perform the expensive E-steps in parallel.
    It assumes that there are 'p' devices, which must EXACTLY match the 'p' in
    the minibatch shape returned by the emissions_loader iterable.

    Currently, this code assumes parallelization over multiple CPU cores on the
    same devices. This requires the user to set the environment flags
        XLA_FLAGS=--xla_force_host_platform_device_count=[p]
    
    Arguments
        initial_params (Parameters)
        prior_params (PriorParams)
        emissions_loader (Generator->[m,t,d] OR -> [p,m,t,d] if parallelize):
            Produces minibatches of emissions. Automatically shuffles after each epoch.
        schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
            schedule; defaults to exponential schedule.
        num_epochs (int): Number of StEM iterations to run over full dataset
        parallelize (bool): If True, parallelize across p devices.
        checkpoint_every (int): Number of (minibatch) iterations between checkpoints;
            an epoch is considered to consist of `num_batches` iterations. If 
            `checkpoint_every` is not a multiple of `num_batches`, additionally
            save the first and last iteration of each epoch. Value must be
            specified to enable checkpointing.
        checkpoint_dir (str): Directory to store checkpoints. Value must be
            specified to enable checkpointing.
        checkpoints_to_keep (int): Number of recent checkpoints to keep.
            Default: -1, keep all checkpoints.
    
    Returns
        fitted_params (Parameters)
        log_probs [num_epochs, num_batches]
    """
    
    num_batches = len(emissions_loader)
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
    if parallelize:
        step_fn = gaussian_hmm.parallel_stochastic_em_step
    else:
        step_fn = gaussian_hmm.nonparallel_stochastic_em_step

    # =========================================================================
    # Initialize / Warm-start
    # =========================================================================
    
    # TODO Why don't we avoid warm-starting in this functioni and allow user
    # to do that in their outer script???
    do_checkpoint = (checkpoint_every is not None) and (checkpoint_dir is not None)
    out = chk.load_latest_checkpoint(checkpoint_dir)
    if do_checkpoint and out[0] is not None:
        print("Warm-starting...", end="")
        ((params, rolling_stats, expected_log_probs, iterator_state), metadata), global_id = out

        emissions_loader.state = iterator_state
        print(f"Loaded checkpoint at step {global_id} from {checkpoint_dir}.")
    else:
        print("Cold-starting from passed-in parameters.")
        global_id = 0

        params = initial_params
        rolling_stats = gaussian_hmm.initialize_statistics(num_states, emission_dim)
        expected_log_probs = []

        metadata = dict(
            prior_params = prior_params,
            schedule     = schedule,
            num_epochs   = num_epochs,
            parallelize  = parallelize,
            num_batches  = num_batches,
            num_states   = num_states,
            emission_dim = emission_dim,
            )

    # =========================================================================
    # Train
    # =========================================================================
    epoch = global_id // num_batches
    minibatch = global_id % num_batches
    
    lp = expected_log_probs[epoch][minibatch] if len(expected_log_probs) > 1 else -jnp.inf
    while epoch < num_epochs:
        expected_log_probs.append([])
        pbar = tqdm(desc=f'epoch {epoch}/{num_epochs}',
                    total=num_batches,
                    initial=minibatch,
                    postfix={'lp': lp})

        for minibatch, minibatch_emissions in enumerate(emissions_loader):
            params, rolling_stats, minibatch_lls = step_fn(
                prior_params, params, rolling_stats, learning_rates[epoch][minibatch],
                minibatch_emissions)

            # Store expected log probability, averaged across total number of emissions
            expected_lp = gaussian_hmm.log_prior(params, prior_params) + num_batches * minibatch_lls
            expected_lp /= emissions_loader.dataset.total_samples
            expected_log_probs[epoch].append(expected_lp)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'lp': expected_lp})
            
            # Save results, if checkpointing
            if do_checkpoint and (global_id % checkpoint_every == 0):    
                tqdm.write(f"Saving checkpoint for epoch {epoch} at {checkpoint_dir}...", end="")
                chk.save_checkpoint(
                    ((params, rolling_stats, expected_log_probs, emissions_loader.state), metadata),
                    global_id,
                    checkpoint_dir,
                    num_checkpoints_to_keep)
                tqdm.write("Done.")

            # Check that M-step emission covariance is PSD
            check_psd(params.emission_covariances)
            
            # Check no NaNs in parameters
            if any(tree_leaves(tree_map(lambda arr: jnp.any(jnp.isnan(arr)), (params, rolling_stats, minibatch_lls)))):
                raise ValueError(f'Epoch {epoch}, minibatch {minibatch}: NaN detected in parameters.')
            
            # Update counter
            global_id += 1
        
        epoch = global_id // num_batches

    return params, jnp.asarray(expected_log_probs)