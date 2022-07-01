"""Fitting and running HMM functions"""

from jax import jit, vmap, pmap
import jax.numpy as np
import jax.random as jr
from functools import partial

from kf.inference import (sharded_e_step, collective_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS,)         
from kf.data_utils import FishPCDataset, FishPCDataloader

from sys import stdout
from tqdm import tqdm

def fit_jmap(train, test, hmm,
             num_iters=10, initial_iter=0, initial_lls=(-np.inf,-np.inf),
             ll_fmt='',
             method='vmap',
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
        method: str, default 'vmap'. The jax parallelization method, either
            'vmap' or 'pmap'. If 'pmap' must ensure number of recognized
            local devices <= batch size outputted by dataloaders.
    """

    # pmap automatically jit-compiles functions, but vmap does not
    jmap = pmap if method=='pmap' else lambda fn: vmap(jit(fn))
    m_step = jit(collective_m_step)

    train_lls = -np.ones(num_iters) * np.inf
    test_lls  = -np.ones(num_iters) * np.inf

    pbar = tqdm(iterable=range(initial_iter, num_iters),
                desc=f"Epochs",
                file=stdout,
                initial=initial_iter,
                postfix=f'train={initial_lls[0]:{ll_fmt}}, test={initial_lls[1]:{ll_fmt}}',)

    # Allocate NGSS. This will be completely updated (in-place) with each e-step
    train_ngss = NGSS.empty((train.num_minibatches, hmm.num_states, hmm.num_obs))
    test_ngss  = NGSS.empty((test.num_minibatches,  hmm.num_states, hmm.num_obs))
    for itr in pbar:
        def e_step(hmm, dl, ngss):
            _e_step = jmap(partial(sharded_e_step, hmm))
            for i_batch, emissions in enumerate(dl):
                bslice = np.s_[i_batch*dl.batch_size:(i_batch+1)*dl.batch_size]
                ngss.batch_set(bslice, _e_step(emissions))
            return

        # Fit on training data
        e_step(hmm, train, train_ngss)
        train_ll = train_ngss.batch_marginal_loglik()
        # hmm = collective_m_step(train_ngss)
        hmm = m_step(train_ngss)
        
        # Evaluate on test data
        e_step(hmm, test, test_ngss)

        # Update progress
        train_lls = train_lls.at[itr].set(train_ngss.batch_marginal_loglik())
        test_lls  = test_lls.at[itr].set(test_ngss.batch_marginal_loglik())

        pbar.set_postfix_str(f'train={train_lls[itr]:{ll_fmt}}, test={test_lls[itr]:{ll_fmt}}')

    return hmm, train_lls, test_lls

fit_vmap = partial(fit_jmap, method='vmap')
fit_pmap = partial(fit_jmap, method='vmap')