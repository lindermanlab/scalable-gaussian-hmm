"""Test FishPCDataset and FishPCLoader classes"""

import os
from pathlib import Path
import numpy as onp

import jax.numpy as jnp
import jax.random as jr

from kf.data import SessionDataset, IteratorState, RandomBatchDataloader

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
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    assert len(concat_dataset.datasets) == num_sessions
    assert len(individual_datasets[0]) == concat_dataset.cumulative_sizes[0]
    for i in range(1, num_sessions):
        assert len(individual_datasets[i]) == (concat_dataset.cumulative_sizes[i] - concat_dataset.cumulative_sizes[i-1])

    idx = jr.randint(sample_key, (), 0, len(concat_dataset))
    data = concat_dataset[idx]
    assert jnp.all(~jnp.isnan(data))
    assert data.shape == concat_dataset.sequence_shape # this might break...

def test_random_batch_dataloader(key=jr.PRNGKey(2205), batch_size=8, seq_length=72_000, seq_step_size=1, num_sessions=10):
    """"""
    seq_key, itr_key = jr.split(key)

    filepaths = FILEPATHS[:num_sessions]
    dataset = \
        SessionDataset.create_multisession(filepaths, seq_key, seq_length, sequence_step_size=1)

    sample_fn = lambda key, ds: jr.permutation(key, len(ds))
    collate_fn = lambda arr_seq: jnp.stack(arr_seq, axis=0)
    iterator_state = IteratorState(key=itr_key, index=jnp.array(0))
    dataloader = RandomBatchDataloader(dataset, batch_size, iterator_state, sample_fn, collate_fn)

    for minibatch, data in enumerate(dataloader):
        if minibatch == 0:
            data_epoch0_iter0 = data.copy()
        assert jnp.all(~jnp.isnan(data))
        assert data.shape == (batch_size, *dataset.datasets[0].sequence_shape)

    # Check that we went through exactly one iteration
    assert minibatch == len(dataloader)-1

    # Check that internal iterator state is being updated
    assert dataloader.iterator_state.index == len(dataloader)

    # Check that on the next iteration, data is shuffled
    for minibatch, data in enumerate(dataloader):
        data_epoch1_iter0 = data.copy()
        break

    assert jnp.all(~jnp.equal(data_epoch0_iter0, data_epoch1_iter0))
    assert dataloader.iterator_state.index == 1
