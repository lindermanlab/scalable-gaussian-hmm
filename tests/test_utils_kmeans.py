"""Test kmeans initialization of GaussianHMM."""

import pytest

import os
import jax.random as jr

from kf import FishPCDataset, kmeans_initialization

DATADIR = os.environ['DATADIR']
fish_name = 'fish0_137'

def make_dataset(num_files=2):
    # Get first num_files days worth of recordings
    fish_dir = os.path.join(DATADIR, fish_name)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])
    filepaths = filepaths[:num_files]

    return FishPCDataset(filepaths, return_labels=False)

def test_kmeans_init(num_states=20, subset_size=72000):
    # Takes about 2.5 min to get 72000 frames and fit.    
    seed = jr.PRNGKey(0)
    ds = make_dataset(2)

    hmm = kmeans_initialization(seed, num_states, ds, subset_size)

    assert hmm.num_states == num_states
    assert hmm.num_obs == ds.dim