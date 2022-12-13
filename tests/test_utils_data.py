"""Test FishPCDataset and FishPCLoader classes"""

import os
import random
import numpy as onp

import jax.random as jr

from kf import (FishPCDataset, FishPCLoader)

DATADIR = os.environ['DATADIR']
fish_name = 'fish0_137'

def make_dataset(num_files=2):
    # Get first num_files days worth of recordings
    fish_dir = os.path.join(DATADIR, fish_name)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])
    filepaths = filepaths[:num_files]
    
    # Create dataset from randomly shuffled filepaths
    # TODO include metadata information such as hatch date
    random.shuffle(filepaths)
    return FishPCDataset(filepaths, return_labels=True)

def test_dataset():
    ds = make_dataset(50)

    # TEST Dataset correctly sorted files and extracted number of frames
    i_files = onp.array([2, 29])
    num_frames = ds.cumulative_frames[i_files] - ds.cumulative_frames[i_files-1]
    assert onp.all(num_frames==[1726618, 1725858])

    # TEST Dataset correctly extracted data for a specific day of recording
    i_file = 42
    f_idx = onp.array([557, 1399, 2297, 3271, 4243, 5297, 6317, 7433])
    ds_idx = f_idx + ds.cumulative_frames[i_file-1]

    # Retrieved from via df.iloc[f_idx, :15], where df is the 42'd file loaded via pandas
    true_arr = onp.array([
        [-1.966646,  0.345493,  0.043362,  0.431899, -0.817103, -0.935952, -0.424814,  0.376971,  0.634887, -0.090397,  0.760033,  0.071908, -0.118746, -0.168039,  0.066190],
        [ 2.615208,  1.025665, -0.066453, -0.477657, -0.482437,  0.861030,  1.129166, -0.180552, -0.757272,  0.098482, -1.435469, -0.230594, -0.243873,  0.693551, -0.407637],
        [-0.902804,  0.334857,  1.022243, -0.166225, -0.576343,  0.244597,  0.479642, -0.327299,  0.698095,  0.089056,  0.248242,  0.241325,  0.407678,  0.138657, -0.235453],
        [14.743798,  3.703990, -2.458721, -1.065586, -0.646491,  0.393797, -1.819442,  3.062200, -0.920465, -0.228458, -1.192837,  3.415640,  0.836353,  4.467017,  0.391304],
        [12.646988,  2.134266, -2.825542, -0.213738, -1.144571, -0.359945,  0.925129,  0.076371, -1.498962,  0.063391, -0.078405,  3.043216, -0.279251,  1.371338, -0.786616],
        [10.047414, -0.259415,  3.082530,  0.382753, -0.576971,  1.417283,  1.518086, -0.198673, -0.973273, -1.250911, -0.740673, -2.170504, -1.460372, -0.503871,  0.235554],
        [ 1.267209,  0.818983,  0.893022, -0.545455,  0.316748,  0.966011, -1.973926, -0.653991,  2.230900,  0.614860, -0.245441,  1.368389,  0.087961,  1.234590, -0.467187],
        [11.345995,  2.287398, -2.228170, -0.294546, -0.225051,  0.383855, -1.956223,  3.434738, -1.336736,  1.779052, -0.474929,  4.321470,  1.996296,  4.421712, -0.477084],
    ])

    arr, labels = ds.collate([ds[idx] for idx in ds_idx])
    assert onp.allclose(arr, true_arr)

    # TODO Check labels
    pass

def test_split_dataset():
    """Split dataset into three sets."""
    ds = make_dataset(2)

    seq_length = 72000   # 1 hr worth of data

    split_sizes = (0.5, 0.2, 0.6) # Adds to more than 1, so last split should be smaller
    seq_slices = ds.split_seq(split_sizes, seed=jr.PRNGKey(0),
                              seq_length=seq_length, drop_incomplete_seqs=True)

    # CHECK Split should return nested lists, outside list same length is num sets
    assert len(seq_slices) == len(split_sizes)
    
    # CHECK split_sizes adds to more than 1, so last set should be smaller than specified
    all_slices = ds.slice_seq(seq_length, drop_incomplete_seqs=True)
    assert len(seq_slices[0]) == int(split_sizes[0] * len(all_slices))
    assert len(seq_slices[1]) == int(split_sizes[1] * len(all_slices))
    assert len(seq_slices[2]) < int(split_sizes[2] * len(all_slices))

def test_dataset_get_frames():
    """Efficiently get individual frames (non-sequential). Used in kmeans init."""
    num_emissions = 100
    ds = make_dataset(3)

    # Get frames from across all files
    idxs = jr.permutation(jr.PRNGKey(0), len(ds))[:num_emissions]
    out = ds.get_frames(idxs)
    assert out.shape == (num_emissions, ds.dim)

    # Get frames from 1st and 3rd file
    idxs = onp.concatenate([
        jr.permutation(jr.PRNGKey(1), ds.cumulative_frames[0])[:num_emissions//2],
        jr.permutation(jr.PRNGKey(2), onp.arange(ds.cumulative_frames[1], ds.cumulative_frames[2]))[:num_emissions//2]
    ])
    out = ds.get_frames(idxs)
    assert out.shape == (num_emissions, ds.dim)

def test_sequential_nonuniform_dataloader():
    """Test DataLoader with no shuffle and incomplete sequence and batch_sizes.
        - Verify dataloader loads correct number of batches per epcoh
        - Verify correct handling of non-uniform batch/sequence lengths
        - verify that dataloder loads batches in exact same sequence each time
    """
    ds = make_dataset(2)
    seq_length = 72000   # 1 hr worth of data
    seq_slices = ds.slice_seq(seq_length, drop_incomplete_seqs=False)
    
    # Create dataloader
    batch_size = 4
    dl = FishPCLoader(ds, seq_slices, batch_size,
                      drop_last=False, shuffle=False)

    # CHECK Correct num batches per epoch, given non-unfirom seq and batch sizes
    assert len(dl) == int(onp.ceil(len(ds) / seq_length / batch_size))

    for i, (batch_data, batch_labels) in enumerate(dl):
        if i==0: # CHECK: Uniform batch size and sequence lengths stacked into array
            assert batch_data.shape == (batch_size, seq_length, 15)
            assert batch_labels.shape == (batch_size, seq_length)

            batch_0 = batch_data.copy() # Save first batch for later comparison
        elif i==5: # CHECK: Not all sequence lengths are uniform
            assert isinstance(batch_data, (list, tuple))
            assert len(batch_data) == batch_size
            assert sum([len(seq) for seq in batch_data]) % seq_length > 0
        elif i==len(dl)-1: # CHECK: Not all batches are full
            assert isinstance(batch_data, (list, tuple))
            assert len(batch_data) <= batch_size
            assert sum([len(seq) for seq in batch_data]) % seq_length > 0

    # CHECK: Ensure batches are loaded in the exact same order
    for i, (batch_data, batch_labels) in enumerate(dl):
        if i==0:
            assert onp.all(batch_data==batch_0)
            break

def test_shuffle_dataloader():
    ds = make_dataset(2)
    seq_length = 72000   # 1 hr worth of data
    seq_slices = ds.slice_seq(seq_length, drop_incomplete_seqs=False)

    batch_size = 4
    dl = FishPCLoader(ds, seq_slices, batch_size,
                      drop_last=True, shuffle=True, seed=459260)
    
    # CHECK Correct num batches per epoch, given uniform seq and batch sizes
    assert len(dl) == len(seq_slices) // batch_size

    for i, (batch_data, batch_labels) in enumerate(dl):
        if i==0: # Uniform batch size and sequence lengths, onp.ndarray
            batch_0 = batch_data.copy() # Save first batch for later comparison
        elif i==len(dl)-1: # CHECK: Final batch is all uniform
            assert batch_data.shape == (batch_size, seq_length, 15)
            assert batch_labels.shape == (batch_size, seq_length)

    # CHECK: Ensure Batches are shuffled with each epoch
    for i, (batch_data, batch_labels) in enumerate(dl):
        if i==0:
            assert not onp.all(batch_data==batch_0)
            break