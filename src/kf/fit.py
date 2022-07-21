"""Fitting and running HMM functions"""

import jax
from jax import jit, vmap, pmap
import jax.numpy as np
import jax.random as jr
from functools import partial

from kf.inference import (streaming_parallel_e_step, SplitBatchOnlineSuffStats)
from kf.data_utils import FishPCDataset, FishPCDataloader

from sys import stdout
from tqdm import tqdm
import chex

def fit_pmap(train, test, hmm,
             num_iters=10, initial_iter=0,
             initial_lls=(-np.inf,-np.inf), ll_fmt='',
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

    train_lls = np.ones(num_iters) * initial_lls[0]
    test_lls  = np.ones(num_iters) * initial_lls[1]

    pbar = tqdm(iterable=range(initial_iter, num_iters),
                desc=f"Epochs",
                file=stdout,
                initial=initial_iter,
                postfix=f'train={initial_lls[0]:{ll_fmt}}, test={initial_lls[1]:{ll_fmt}}',)
    
    num_devices = jax.local_device_count()
    get_num_emissions = lambda dl: len(dl) * num_devices \
                                   * (dl.batch_shape[0]//num_devices) \
                                   * dl.num_frames_per_batch

    for itr in pbar:

        # Fit on training data
        train_suff_stats = streaming_parallel_e_step(hmm, train)
        train_suff_stats = jax.tree_map(lambda arr: np.expand_dims(arr, axis=0), train_suff_stats)
        hmm.m_step(None, train_suff_stats)

        train_ll = train_suff_stats.marginal_loglik.squeeze() / get_num_emissions(train)
        
        # Evaluate on test data
        test_suff_stats = streaming_parallel_e_step(hmm, test)
        test_ll = test_suff_stats.marginal_loglik.squeeze() / get_num_emissions(test)

        # Update progress
        train_lls = train_lls.at[itr].set(train_ll)
        test_lls  = test_lls.at[itr].set(test_ll)

        pbar.set_postfix_str(f'train={train_lls[itr]:{ll_fmt}}, test={test_lls[itr]:{ll_fmt}}')

    return hmm, train_lls, test_lls