"""Test FishPCDataset and FishPCLoader classes"""

import os
from pathlib import Path
import numpy as onp

import jax.numpy as jnp
import jax.random as jr

from kf.data import SessionDataset

DATADIR = Path(os.environ['DATADIR'])
FILEPATHS = filepaths = sorted((DATADIR/'fish0_137').glob('*.h5'))

def test_single_session(seq_key=jr.PRNGKey(0), seq_length=72_000, seq_step_size=1):
    """Check shapes and __get__item of single session dataset."""
    filepath = FILEPATHS[0]
    dataset = SessionDataset(filepath, seq_key, seq_length, seq_step_size)

    assert dataset.raw_shape[0] // (seq_length*seq_step_size) == len(dataset)

def test_multi_session(seq_key=jr.PRNGKey(0), seq_length=72_000, seq_step_size=1, num_sessions=10):
    """Check shapes and __get__item of single session dataset."""
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