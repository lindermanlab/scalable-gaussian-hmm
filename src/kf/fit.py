"""Fitting and running HMM functions"""

from sys import stdout
from tqdm import tqdm
import chex

import jax
import jax.numpy as jnp
import jax.random as jr

from kf.inference import streaming_parallel_e_step
from kf.data_utils import FishPCDataset, FishPCDataloader

from memory_profiler import profile
import gc

@profile
def fit_pmap(train, test, hmm,
             num_iters=10, initial_iter=0,
             initial_lls=(-jnp.inf,-jnp.inf), ll_fmt='',
            ):
    """Fit HMM to principle component data by [v/p]mapping over batches using EM.

    Parameters
        train: FishPCDataloader
        test: FishPCDataloader
        hmm: GaussianHMM
        num_iters: int, default 10. Number of EM iterations to run.
        initial_iter: int, default 0. Intial EM iteration counter value.
            Useful when warm-starting from a saved HMM.
        initial_lls: tuple of floats, default (-inf, -inf). Initial train and
            test log-likelihoods. Useful when warm-starting from a saved HMM.
        ll_fmt: pyformat style str (e.g. :.2f)
    """
    train_lls = jnp.ones(num_iters) * initial_lls[0]
    test_lls  = jnp.ones(num_iters) * initial_lls[1]

    pbar = tqdm(iterable=range(initial_iter, num_iters),
                desc=f"Epochs",
                file=stdout,
                initial=initial_iter,
                postfix=f'train={initial_lls[0]:{ll_fmt}}, test={initial_lls[1]:{ll_fmt}}',)
    
    num_devices = jax.local_device_count()
    get_num_emissions = lambda dl: len(dl) * num_devices \
                                   * (dl.batch_shape[0]//num_devices) \
                                   * dl.num_frames_per_batch

    n_batch_train = get_num_emissions(train)
    n_batch_test  = get_num_emissions(test)

    for itr in pbar:

        # Fit on training data
        train_suff_stats = streaming_parallel_e_step(hmm, train)
        train_ll = train_suff_stats.marginal_loglik.sum() / n_batch_train

        train_suff_stats = jax.tree_map(lambda arr: jnp.expand_dims(arr, 0), train_suff_stats)
        hmm.m_step(None, train_suff_stats)
        
        # Evaluate on test data
        test_suff_stats = streaming_parallel_e_step(hmm, test)
        test_ll = test_suff_stats.marginal_loglik.sum() / n_batch_test 

        # Update progress
        train_lls = train_lls.at[itr].set(train_ll)
        test_lls  = test_lls.at[itr].set(test_ll)

        pbar.set_postfix_str(f'train={train_lls[itr]:{ll_fmt}}, test={test_lls[itr]:{ll_fmt}}')

        del train_suff_stats, test_suff_stats

    return hmm, train_lls, test_lls