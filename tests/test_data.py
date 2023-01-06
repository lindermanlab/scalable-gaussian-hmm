"""Test FishPCDataset and FishPCLoader classes"""

import os
from pathlib import Path
import numpy as onp

import jax.numpy as jnp
import jax.random as jr

from kf.data import (SingleSessionDataset,
                     MultiSessionDataset,
                     RandomBatchSampler,
                     get_file_raw_shapes)
from torch.utils.data import DataLoader

DATADIR = Path(os.environ['DATADIR'])
FILEPATHS = sorted((DATADIR/'fish0_137').glob('*.h5'))
FILEPATHS, _ = get_file_raw_shapes(FILEPATHS, min_frames_per_file=72_000)

def test_single(key=jr.PRNGKey(0), seq_length=1_000, seq_step_size=1):
    """Check shapes and __get__item of single session dataset."""
    seq_key, sample_key = jr.split(key)

    filepath = FILEPATHS[0]
    dataset = SingleSessionDataset(filepath, seq_key, seq_length, seq_step_size)

    assert dataset.raw_shape[0] // (seq_length*seq_step_size) == len(dataset)
    
    idx = jr.randint(sample_key, (), 0, len(dataset))
    data = dataset[idx]
    assert jnp.all(~jnp.isnan(data))
    assert data.shape == dataset.sequence_shape

def test_multi(key=jr.PRNGKey(0), seq_length=72_000, seq_step_size=1, num_sessions=10):
    """Check shapes and __get__item of dataset consisting of multiple sessions."""
    seq_key, sample_key = jr.split(key)

    num_sessions = jnp.maximum(3,10)
    filepaths = FILEPATHS[:num_sessions]
    individual_datasets = [SingleSessionDataset(f, k, seq_length, seq_step_size)
                           for f, k in zip(filepaths, jr.split(seq_key, len(filepaths)))]
    multi_dataset = \
        MultiSessionDataset(filepaths, seq_key, seq_length, seq_step_size)

    assert len(multi_dataset.datasets) == num_sessions
    assert len(individual_datasets[0]) == multi_dataset.cumulative_sizes[0]
    for i in range(1, num_sessions):
        assert len(individual_datasets[i]) == (multi_dataset.cumulative_sizes[i] - multi_dataset.cumulative_sizes[i-1])

    idx = jr.randint(sample_key, (), 0, len(multi_dataset))
    data = multi_dataset[idx]
    assert jnp.all(~jnp.isnan(data))
    assert data.shape == multi_dataset.sequence_shape # this might break...

def test_dataloader(key=jr.PRNGKey(2205), batch_size=8, seq_length=72_000, num_sessions=10):
    """Test auto-shuffle with each epoch"""
    seq_key, itr_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = MultiSessionDataset(filepaths, seq_key, seq_length)

    collate_fn = lambda arr_seq: jnp.stack(arr_seq, axis=0)
    sampler = RandomBatchSampler(dataset, batch_size, itr_key)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    for minibatch, data in enumerate(dataloader):
        if minibatch == 0:
            data_epoch0_iter0 = data.copy()
        assert jnp.all(~jnp.isnan(data))
        assert data.shape == (batch_size, *dataset.datasets[0].sequence_shape)

    # Check that we went through exactly one iteration
    assert minibatch == len(dataloader)-1

    # Check that internal iterator state is being updated
    assert dataloader.batch_sampler.index == len(dataloader)

    # Check that on the next iteration, data is shuffled
    for minibatch, data in enumerate(dataloader):
        data_epoch1_iter0 = data.copy()
        break

    assert jnp.all(~jnp.equal(data_epoch0_iter0, data_epoch1_iter0))
    assert dataloader.batch_sampler.index == 1

def test_dataloader_warm(key=jr.PRNGKey(2205), batch_size=8, seq_length=72_000, num_sessions=10):
    """Test warm-start capability"""
    i_warm = 4

    seq_key, itr_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = MultiSessionDataset(filepaths, seq_key, seq_length)

    collate_fn = lambda arr_seq: jnp.stack(arr_seq, axis=0)
    
    sampler_cold = RandomBatchSampler(dataset, batch_size, itr_key)

    sampler_warm = RandomBatchSampler(dataset, batch_size, itr_key)
    sampler_warm.state = {'key': itr_key, 'index': i_warm}
    
    for minibatch, data in enumerate(DataLoader(dataset, batch_sampler=sampler_cold, collate_fn=collate_fn)):
        if minibatch == i_warm:
            data_cold = data.copy()
        assert jnp.all(~jnp.isnan(data))
        assert data.shape == (batch_size, *dataset.datasets[0].sequence_shape)

    # Check that the first loaded data batch is indeed the warm-started one
    for data in DataLoader(dataset, batch_sampler=sampler_warm, collate_fn=collate_fn):
        data_warm = data.copy()
        break

    assert jnp.all(jnp.equal(data_cold, data_warm))