"""Time profile different variations of data subsetting and k-means."""

import os
import argparse
from line_profiler import LineProfiler

import jax.random as jr
import numpy as onp
from sklearn.cluster import KMeans
from kf import FishPCDataset

parser = argparse.ArgumentParser(description='Time profile of various implementations of k-means')
parser.add_argument(
    '--method', type=str, default='individual',
    choices=['individual', 'interval'],
    help='K-means method')
parser.add_argument(
    '--seed', type=int, default=224815,
    help='Initial RNG seed, for splitting data and intializing HMM.')
parser.add_argument(
    '--clusters', type=int, default=20,
    help='Number of clusters to fit')
parser.add_argument(
    '--size', type=float, default=25000,
    help='Number of samples to fit. If >1, interpret as an integer. If [0, 1], interpreted as fraction of dataset.')

def kmeans_sklearn_individual(seed, n_clusters, dataset, subset_size=0.001):
    """Randomly select N datapoints from dataset.
    
    Args:
        seed (jr.PRNGKey):
        n_clusters (int): Number of clusters to fit
        dataset (torch.utils.data.Dataset):
        subset_size (float): Size of dataset to use in kmeans fit. If >1, value
            is interpreted as number of samples. If (0, 1], value is interpreted
            as fraction of dataset. Note the difference in how a value of 1 is
            interpreted here, vs. e.g. FishPCDataset.split_seq. Default: 0.001.
    """

    print('In kmeans_sklearn_individual...')

    seed_data, seed_kmeans = jr.split(seed)

    # Get subset of data from dataset
    num_samples = int(len(dataset) * subset_size) \
                  if subset_size < 1 else int(subset_size)
    idxs = jr.permutation(seed_data, len(dataset))[:num_samples]

    emissions = dataset.get_frames(idxs)

    print(f'\tLoaded {len(emissions)} samples')

    # Set emission means and covariances based on fitted k-means clusters
    kmeans = KMeans(n_clusters, random_state=int(seed_kmeans[-1])).fit(emissions)

    return kmeans

def kmeans_sklearn_interval(seed, n_clusters, dataset, subset_size=0.001,
                            seq_length=250, step_size=6000):
    """Lloyd's batch k-means using sklearn implementation. Selects datasets at
    a set an interval (e.g. 2 minutes, such that sufficiently different states
    are being sampled.)
    
    Args:
        seed (jr.PRNGKey):
        n_clusters (int): Number of clusters to fit
        dataset (torch.utils.data.Dataset):
        subset_size (float): Size of dataset to use in kmeans fit. If >1, value
            is interpreted as number of samples. If (0, 1], value is interpreted
            as fraction of dataset. Note the difference in how a value of 1 is
            interpreted here, vs. e.g. FishPCDataset.split_seq. Default: 0.001.
        step_size (int): Number of frames between consecutive frames in a sequence.
            If too small, the selcted data will all be similar. Default: 6000,
            corresponding to 5 minutes @ 20 Hz.
        seq_length (int): Number of frames in a sequence. If too small, a lot of
            time may be spent accessing data. If too large, a lot of data given
            to algorithm will be very similar. Default: With step_size, set so
            that ~< 1 full day 
    """

    print('In kmeans_sklearn_interval...')

    seed_data, seed_kmeans = jr.split(seed)

    # Get subset of data from dataset
    # seq_length == num_frames_in_seq
    # N sets of tuples....
    seq_slices = dataset.slice_seq(seq_length, step_size=step_size, drop_incomplete_seqs=True)

    num_samples = int(len(dataset) * subset_size) \
                  if subset_size < 1 else int(subset_size)
    num_seqs = num_samples // seq_length

    seq_idxs = jr.permutation(seed_data, len(seq_slices))[:num_seqs]

    emissions = onp.stack([dataset[seq_slices[idx]] for idx in seq_idxs], axis=0)
    emissions = emissions.reshape(-1, 15)

    print(f'\tLoaded {len(emissions)} samples')

    # Set emission means and covariances based on fitting k-means clusters
    kmeans = KMeans(n_clusters, random_state=int(seed_kmeans[-1])).fit(emissions)

    return kmeans

if __name__ == '__main__':
    args = parser.parse_args()

    method = args.method

    seed = jr.PRNGKey(args.seed)
    n_clusters = args.clusters
    subset_size = args.size

    # Setup dataset
    DATADIR = os.environ['DATADIR']
    fish_id = 'fish0_137'
    fish_dir = os.path.join(DATADIR, fish_id)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])

    # FIXME Hard limiting number of filepaths. Remove when done debugging
    filepaths = filepaths[:10]
    print(f"WARNING: FORCING DATASET TO ONLY LOAD {len(filepaths)} files of data")
    dataset = FishPCDataset(filepaths, return_labels=False)

    # Run
    profile = LineProfiler()
    args = (seed, n_clusters, dataset, subset_size)
    if method == 'interval':
        fn = profile(kmeans_sklearn_interval)
        kwargs = {'seq_length': 250, 'step_size': 6000}
        fn(*args, **kwargs)
    else:
        fn = profile(kmeans_sklearn_individual)
        kwargs = {}
        fn(*args, **kwargs)

    profile.print_stats()