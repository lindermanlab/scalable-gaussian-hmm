"""Test FishPCDataset and FishPCLoader classes"""

import os
from pathlib import Path
import numpy as onp

import jax.numpy as jnp
import jax.random as jr

from kf.data import (SessionDataset,
                     MultisessionDataset,
                     random_split,
                     IteratorState,
                     RandomBatchDataloader)

DATADIR = Path(os.environ['DATADIR'])
FILEPATHS = filepaths = sorted((DATADIR/'fish0_137').glob('*.h5'))

def test_single_session(key=jr.PRNGKey(0), seq_length=72_000, seq_step_size=1):
    """Check shapes and __get__item of single session dataset."""
    seq_key, sample_key = jr.split(key)

    filepath = FILEPATHS[0]
    dataset = SessionDataset(filepath, seq_key, seq_length, seq_step_size)

    assert dataset.raw_shape[0] // (seq_length*seq_step_size) == len(dataset)
    
    idx = jr.randint(sample_key, (), 0, len(dataset))
    data = dataset[idx]
    assert jnp.all(~jnp.isnan(data))
    assert data.shape == dataset.sequence_shape

def test_multi_session(key=jr.PRNGKey(0), seq_length=72_000, seq_step_size=1, num_sessions=10):
    """Check shapes and __get__item of dataset consisting of multiple sessions."""
    seq_key, sample_key = jr.split(key)

    num_sessions = jnp.maximum(3,10)
    filepaths = FILEPATHS[:num_sessions]
    individual_datasets = [SessionDataset(f, k, seq_length, seq_step_size)
                           for f, k in zip(filepaths, jr.split(seq_key, len(filepaths)))]
    concat_dataset = \
        MultisessionDataset.init_from_paths(filepaths, seq_key, seq_length, sequence_step_size=1)

    assert len(concat_dataset.datasets) == num_sessions
    assert len(individual_datasets[0]) == concat_dataset.cumulative_sizes[0]
    for i in range(1, num_sessions):
        assert len(individual_datasets[i]) == (concat_dataset.cumulative_sizes[i] - concat_dataset.cumulative_sizes[i-1])

    idx = jr.randint(sample_key, (), 0, len(concat_dataset))
    data = concat_dataset[idx]
    assert jnp.all(~jnp.isnan(data))
    assert data.shape == concat_dataset.sequence_shape # this might break...

def test_dataloader(key=jr.PRNGKey(2205), batch_size=8, seq_length=72_000, seq_step_size=1, num_sessions=10):
    """Test auto-shuffle with each epoch"""
    seq_key, itr_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = \
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    sample_fn = lambda key, ds: jr.permutation(key, len(ds))
    collate_fn = lambda arr_seq: jnp.stack(arr_seq, axis=0)
    dataloader = RandomBatchDataloader(dataset, batch_size, itr_key, sample_fn, collate_fn)

    for minibatch, data in enumerate(dataloader):
        if minibatch == 0:
            data_epoch0_iter0 = data.copy()
        assert jnp.all(~jnp.isnan(data))
        assert data.shape == (batch_size, *dataset.datasets[0].sequence_shape)

    # Check that we went through exactly one iteration
    assert minibatch == len(dataloader)-1

    # Check that internal iterator state is being updated
    assert dataloader.index == len(dataloader)

    # Check that on the next iteration, data is shuffled
    for minibatch, data in enumerate(dataloader):
        data_epoch1_iter0 = data.copy()
        break

    assert jnp.all(~jnp.equal(data_epoch0_iter0, data_epoch1_iter0))
    assert dataloader.index == 1

def test_dataloader_warm(key=jr.PRNGKey(2205), batch_size=8, seq_length=72_000, num_sessions=10):
    """Test warm-start capability"""
    i_warm = 4

    seq_key, itr_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = \
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    sample_fn = lambda key, ds: jr.permutation(key, len(ds))
    collate_fn = lambda arr_seq: jnp.stack(arr_seq, axis=0)
    dataloader_cold = RandomBatchDataloader(dataset, batch_size, itr_key, sample_fn, collate_fn)

    dataloader_warm = RandomBatchDataloader(dataset, batch_size, itr_key, sample_fn, collate_fn)
    dataloader_warm.state = IteratorState(key=itr_key, index=i_warm)

    for minibatch, data in enumerate(dataloader_cold):
        if minibatch == i_warm:
            data_cold = data.copy()
        assert jnp.all(~jnp.isnan(data))
        assert data.shape == (batch_size, *dataset.datasets[0].sequence_shape)

    # Check that the first loaded data batch is indeed the warm-started one
    for data in dataloader_warm:
        data_warm = data.copy()
        break

    assert jnp.all(jnp.equal(data_cold, data_warm))

def test_split_full(key=jr.PRNGKey(0), seq_length=3600, num_sessions=3):
    """Split dataset into three sets."""

    seq_key, split_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = \
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    # Split dataset
    split_sizes = (0.5, 0.2, 0.3) # Adds to more than 1, so last split should be smaller
    split_datasets = random_split(split_key, dataset, split_sizes)

    # Split should return list of SubDatasets
    assert len(split_sizes) == len(split_datasets)
    
    # Since split_sizes equals 1 exactly, then all samples in dataset should be used
    split_ds_sizes = [len(ds) for ds in split_datasets]
    assert sum(split_ds_sizes) == len(dataset)

    # Check that we can read data from each dataset
    assert jnp.all(~jnp.isnan(split_datasets[1][304]))

def test_split_partial(key=jr.PRNGKey(0), seq_length=3600, num_sessions=3):
    """Split dataset into two partial sets."""

    seq_key, split_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = \
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    # Split dataset
    split_sizes = (0.2, 0.4)
    split_datasets = random_split(split_key, dataset, split_sizes)

    # Split should return list of SubDatasets
    assert len(split_sizes) == len(split_datasets)
    
    # Since split_sizes equals 1 exactly, then all samples in dataset should be used
    split_ds_sizes = [len(ds) for ds in split_datasets]
    assert jnp.isclose(sum(split_ds_sizes), sum(split_sizes)*len(dataset), atol=len(split_sizes)*1e0)

    # Check that we can read data from each dataset
    assert jnp.all(~jnp.isnan(split_datasets[0][230]))