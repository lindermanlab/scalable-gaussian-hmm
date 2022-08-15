import pytest

import os
import tempfile
import random

import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from kf import (CheckpointDataclass, train_and_checkpoint)
from ssm_jax.hmm.models import GaussianHMM
from torch.utils.data import DataLoader

def make_rnd_model_and_data(num_states=5, emission_dim=2,
                            num_timesteps=2000, num_batches=1,
                            seed=jr.PRNGKey(0),):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

    # Generate data from the model
    batch_true_states, batch_emissions = \
        vmap(true_hmm.sample, in_axes=(0, None))\
            (jr.split(seed, num_batches), num_timesteps)

    return true_hmm, batch_true_states, batch_emissions

def test_checkpoint_save(num_files=2, num_epochs=10):
    """Verify that HMM loaded from file exactly matches original HMM."""
    
    init_key = jr.PRNGKey(3420493)
    num_states, num_obs = 5, 2
    original_hmm = GaussianHMM.random_initialization(init_key, num_states, num_obs)

    # Save HMM using Checkpoint functionality
    tempdir = tempfile.TemporaryDirectory()
    checkpoint = CheckpointDataclass(directory=tempdir.name, prefix='tmp_', interval=1,)
    ckp_path = checkpoint.save(original_hmm, 0, all_lps=[])
    
    # Load the saved HMM, ensure that all unconstrained params are equal
    loaded_hmm, epochs_completed, all_lps = checkpoint.load(ckp_path)

    assert all(
        [onp.all(og==ld) for og, ld in zip(original_hmm.unconstrained_params, loaded_hmm.unconstrained_params)]
    ), 'Expected unconstrained params of HMM loaded from file to match original HMM.'

class ArrayLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=True):
        
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: jnp.array(batch),
            drop_last=drop_last,
            )

        self.total_emissions = len(self) * batch_size * dataset.shape[-2]
        return

def test_warmstart(num_files=2):
    """Verify that HMM run with checkpointing matches HMM trained in one go.
    Results will not exactly match because of how Dataloader shuffles data.
    """
    key = jr.PRNGKey(3420493)
    data_key, init_key = jr.split(key, 2)

    seq_length = 72000   # 1 hr worth of data
    batch_size = 4

    num_states, num_obs = 5, 2
    _, _, batch_emissions = make_rnd_model_and_data(
        num_states=num_states, emission_dim=num_obs,
        num_timesteps=2000, num_batches=4,
        seed=data_key
    )

    dl = ArrayLoader(batch_emissions)

    # Initialize HMMs identically
    total_num_epochs = 10
    checkpoint_interval = 2
    refr_hmm = GaussianHMM.random_initialization(init_key, num_states, num_obs)
    test_hmm = GaussianHMM.random_initialization(init_key, num_states, num_obs)

    # Force partial-training (6/10 epochs) for one HMM, with intermediate checkpointing.
    # This should produce 3 checkpoint files.
    partial_epochs = 6

    tempdir = tempfile.TemporaryDirectory()
    checkpoint = CheckpointDataclass(directory=tempdir.name, prefix='tmp_', interval=checkpoint_interval)

    _, last_ckp_path = train_and_checkpoint(dl, test_hmm, partial_epochs, checkpoint)
    
    assert len(os.listdir(checkpoint.directory)) == partial_epochs // checkpoint.interval, \
        f'Expected {partial_epochs // checkpoint.interval} checkpoint files to be created.'
    
    # Train for remainder of epochs, with warm-started HMMContinue training, with reloaded HMM
    test_hmm, epochs_completed, partial_lps = checkpoint.load(last_ckp_path)
    remainder_lps, _ = train_and_checkpoint(
        dl, test_hmm, total_num_epochs, checkpoint, starting_epoch=epochs_completed+1
    )

    test_lps = jnp.concatenate([partial_lps, remainder_lps])

    assert len(test_lps) == total_num_epochs
    
    # TEST Parameter values and lps are similiar between test and reference HMM
    # Since we are using the same DataLoader instance, the two HMMs will be
    # training on different data. So, results will not be identical.
    refr_lps = refr_hmm.fit_stochastic_em(dl, dl.total_emissions, num_epochs=total_num_epochs)

    assert jnp.allclose(test_lps[-1], refr_lps[-1], atol=100)
    assert jnp.allclose(test_hmm.emission_means.value,
                        refr_hmm.emission_means.value,
                        atol=1)
    assert jnp.allclose(test_hmm.emission_covariance_matrices.value,
                        refr_hmm.emission_covariance_matrices.value,
                        atol=1)