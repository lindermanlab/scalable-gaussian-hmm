"""Evaluate k-means cluster validity of interval approach."""

import os
import argparse
import time

from jax import vmap
import jax.random as jr
import numpy as onp
from sklearn.cluster import KMeans

from kf import FishPCDataset

NUM_FILES = 36                  # 30 files ~= 1 file / 1 weeks
FPATH = os.getenv('DATADIR')

parser = argparse.ArgumentParser(description='Cluster validity')
parser.add_argument(
    '--run', type=str, default=None,
    choices=['ref', 'test', 'eval'],
    help='Reference vs. test approach')
parser.add_argument(
    '--seed', type=int, default=22290816,
    help='Initial RNG seed, for splitting data and intializing HMM.')
parser.add_argument(
    '--clusters', type=int, default=20,
    help='Number of clusters to fit')
parser.add_argument(
    '--size', type=float, default=25000,
    help='Number of samples to fit. If >1, interpret as an integer. If [0, 1], interpreted as fraction of dataset.')

def make_reference_data(seed, all_filepaths, step_size=1200, seq_length=180):
    """

    Default values of step_size=2400, seq_length=300 result in:
        3 hrs of data per sequence @ 1 sample every 1 min
    """

    # Randomly selected NUM_FILES uniformly from lifespan
    idxs = jr.permutation(seed, len(all_filepaths))[:NUM_FILES]
    filepaths = [all_filepaths[i] for i in idxs]
    dataset = FishPCDataset(filepaths, return_labels=False, min_frames=seq_length*step_size)

    # Define fine grid of seq slices and choose
    seq_slices = dataset.slice_seq(seq_length, step_size, drop_incomplete_seqs=True)

    emissions = onp.stack([dataset[slc] for slc in seq_slices], axis=0)
    emissions = emissions.reshape(-1, 15)

    return emissions

def get_random_slices(seed, dataset, num_seqs, seq_length, step_size):
    """Return the list of random index tuples to slice into data sequentially for
    given sequence length and step size. Automatically drops incomplete sequences.
    
    Each index tuple produces values in the interval [start, stop) where
        seq_length = (stop - start) * step_size.
    
    Args:
        seed (jr.PRNGKey): PRNG seed
        dataset:
        num_seqs (int)
        seq_length (int): Number of consecutive frames in a sequence
        step_size (int): Spacing between consecutive frames

    Returns:
        slices (list): List of tuples consist of (start, stop, step) values
    """

    seed_day, seed_frame = jr.split(seed)
    
    abs_seq_length = seq_length * step_size

    # Randomly choose files to get data. Files may be selected more than once.    
    days = jr.randint(seed_day, (num_seqs,), 0, len(dataset.cumulative_frames))

    # Randomly choose a start time with the first (num_frames_of_file - abs_seq_length)
    # frames for each day. This (should) ensures that all sequences are complete.
    start_range_per_file = dataset._num_frames_per_file[days] - abs_seq_length
    start_frames = vmap(jr.randint, in_axes=(0,None,None,0)) \
                       (jr.split(seed_frame, num_seqs), (), 0, start_range_per_file)
    start_frames = onp.array(start_frames)

    # Then, correct start_frames index to encode which day they came from
    start_frames += onp.asarray([0 if d == 0 else dataset.cumulative_frames[d-1] for d in days])

    return list(zip(start_frames, start_frames + abs_seq_length, onp.ones(num_seqs)*step_size))

def make_interval_data(seed, all_filepaths, num_samples, step_size=2400, seq_length=120):
    """Get emissions from randomly interval-sliced dataset.
 
    Default values of subset_size=0.001, step_size=6000, seq_length=125 result in:
        4 hrs of data per sequence @ 1 sample every 2 minutes, XXX samples
    Args
        seed (jr.PRNGKey): RNG seed
        all_filepaths (list): Files to load data from
        num_samples (float): Target number of samples to fit kmeans over.
            interpreted here, vs. e.g. FishPCDataset.split_seq. Default: 0.001.
        step_size (int): Number of frames between consecutive frames in a sequence.
            If too small, the selcted data will all be similar. Default: 6000,
            corresponding to 5 minutes @ 20 Hz.
        seq_length (int): Number of frames in a sequence. If too small, a lot of
            time may be spent accessing data. If too large, a lot of data given
            to algorithm will be very similar. Default: With step_size, set so
            that ~< 1 full day 
    """
    
    # Create dataset from all long-enough files
    abs_seq_length = seq_length * step_size
    dataset = FishPCDataset(all_filepaths, return_labels=False, min_frames=abs_seq_length)

    # Randomly get slices of specified step size and length
    num_seqs = num_samples // seq_length
    seq_slices = get_random_slices(seed, dataset, num_seqs, seq_length, step_size)
    
    emissions = onp.stack([dataset[slc] for slc in seq_slices], axis=0)
    emissions = emissions.reshape(-1, 15)

    return emissions

if __name__ == '__main__':
    args = parser.parse_args()

    if args.run == 'eval':
        raise NotImplementedError

    elif args.run =='ref' or args.run == 'test':
        seed_data, seed_kmeans = jr.split(jr.PRNGKey(args.seed))
        n_clusters = args.clusters
        # subset_size = args.size
        
        num_samples = 35280 # subset_size = 0.0001 of most filepaths
        print(f'Attempting to load {num_samples} samples...')

        DATADIR = os.environ['DATADIR']
        fish_id = 'fish0_137'
        fish_dir = os.path.join(DATADIR, fish_id)
        filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])

        # -------------------
        tic = time.time()
        if args.run == 'ref':
            fpath = os.path.join(DATADIR, 'validate_cluster-ref.npz')
            emissions = make_reference_data(seed_data, filepaths)

        elif args.run == 'test':
            # RESULTS: 352800 samples, 1.62 min
            fpath = os.path.join(DATADIR, 'validate_cluster-interval.npz')
            emissions = make_interval_data(seed_data, filepaths, num_samples)

        # Run K-means and save results
        print(f'\tLoaded {len(emissions)} samples. Running k-means...')
        kmeans = KMeans(n_clusters, random_state=int(seed_kmeans[-1])).fit(emissions)
        toc = time.time()
        # -------------------

        onp.savez(fpath,
                cluster_means=kmeans.cluster_centers_,
                cluster_label=kmeans.labels_,
                fit_objective=kmeans.inertia_,
        )

        print(f'Elapsed run time: {(toc-tic)/60.:.2f} min')
        print(f'K-means results saved to {fpath}')