"""Print number of batches and batch shapes of resulting Dataloader oterators."""

import os
import argparse

from kf.data_utils import FishPCDataset, FishPCDataloader

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']
fish_id = 'fish0_137'

# -------------------------------------

parser = argparse.ArgumentParser(description='Estimate batch shapes')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Number of batches loaded per iteration.')
parser.add_argument(
    '--frames_per_batch', type=int, default=100000,
    help='Number of frames per batch. Also used to filter out files which do not have this many frames.')
parser.add_argument(
    '--max_frames_per_file', type=int, default=-1,
    help='Maximum number of frames available in each file/day. Useful for debugging.')
parser.add_argument(
    '--num_train', type=float, default=1,
    help='Fraction of dataset to train over')
parser.add_argument(
    '--num_test', type=float, default=1,
    help='Fraction of dataset to test over')
parser.add_argument(
    '--seed', type=int, default=None,
    help='PRNG seed to split data and intialize HMM.')

def main():
    args = parser.parse_args()
    
    batch_size = args.batch_size
    frames_per_batch = args.frames_per_batch
    max_frames_per_file = args.max_frames_per_file
    num_train = args.num_train
    num_test = args.num_test
    seed = args.seed

    if seed is not None:
        import jax.random as jr
        seed_split_data, _ = jr.split(seed, 2)
    else:
        seed_split_data = None

    # ==========================================================================
    full_ds = FishPCDataset(fish_id, DATADIR,
                            data_subset='all',
                            min_frames_per_file=frames_per_batch,
                            max_frames_per_file=max_frames_per_file) # used for debugging
    train_ds, test_ds = full_ds.train_test_split(num_train=num_train,
                                                 num_test=num_test,
                                                 seed=seed_split_data,)
    del full_ds

    # TODO Move train_test_split function to FishPCDataloader class
    # so that we can shuffle over batches (and not just over days)
    # Load all emissions, shape (num_days, frames_per_batch, dim)
    train_dl = FishPCDataloader(train_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=frames_per_batch)
          
    test_dl  = FishPCDataloader(test_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=frames_per_batch)
    
    def fmtstr(name, value, value_fmt='', padding=25):
        return f"   {name+':  ':>{padding}}{value:{value_fmt}}"

    print(f"{'Parameters ':-<50}")
    print(fmtstr("batch_size", batch_size))
    print(fmtstr("frames_per_batch", frames_per_batch))
    print(fmtstr("max_frames_per_file", max_frames_per_file))

    print(fmtstr("num_train", num_train, '.3f')
          +f", corresponding to {len(train_ds):3} files")
    print(fmtstr("num_test", num_test, '.3f')
          +f", corresponding to {len(test_ds):3} files")
    
    print(f"\n{'Results ':-<50}")
    print(f"  Train: {len(train_dl):3d} batches of shape {train_dl.batch_shape}")
    print(f"   Test: {len(test_dl):3d} batches of shape {test_dl.batch_shape}")
    print()
    return

if __name__ == '__main__':
    main()