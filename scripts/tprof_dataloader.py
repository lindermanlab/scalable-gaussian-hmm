"""Profile time to load all data from one fish's lifespan.

Time may be a function of batch sizesa and sequence lengths.
"""

import os
import sys
import argparse
from datetime import datetime
import time

import numpy as onp
from tqdm import trange, tqdm
from tqdm.contrib.itertools import product

from kf import FishPCDataset, FishPCLoader

TEMPDIR = os.environ['TEMPDIR']
DATADIR = os.environ['DATADIR']
fish_id = 'fish0_137'

parser = argparse.ArgumentParser(description='Profile time usage of loading data')
parser.add_argument(
    '--log_dir', type=str, default=TEMPDIR,
    help='Directory to log profiles.')
parser.add_argument(
    '--session_name', type=str, default=None,
    help='Identifying token,. Used for log and checkpoint files')

def setup_data(batch_size, seq_length):

    fish_dir = os.path.join(DATADIR, fish_id)
    filepaths = sorted([os.path.join(fish_dir, f) for f in os.listdir(fish_dir)])

    dataset = FishPCDataset(filepaths)

    all_slices = dataset.slice_seq(seq_length, step_size=1, drop_incomplete_seqs=True)
    dataloader = FishPCLoader(dataset, all_slices, batch_size, drop_last=False, shuffle=True)
        
    return dataset, dataloader

def main():
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    session_name = args.session_name if args.session_name is not None else timestamp

    batch_sizes = [4, 16, 64]
    seq_lengths = [864000, 432000, 72000, 36000, 18000] # 12hrs, 6hrs, 1hr, 0.5, 0.25 hr worth of data
    
    pbar = tqdm(product(batch_sizes, seq_lengths),
                file=sys.stdout,
                postfix={'batch_size': 'n/a', 'seq_length': 'n/a', 'n_batches': 'n/a'})

    results = onp.empty((len(batch_sizes)*len(seq_lengths), 4))
    for i, (batch_size, seq_length) in enumerate(pbar):
        _, dataloader = setup_data(batch_size, seq_length)
        pbar.set_postfix({'batch_size': batch_size,
                          'seq_length': f"{seq_length//1000:3d}K",
                          'n_batches': len(dataloader)})

        # ===========================================
        tic = time.perf_counter()
        for emission_label_pairs in dataloader:
            _ = emission_label_pairs[0][0,...] + emission_label_pairs[0][-1,...]
        toc = time.perf_counter()
        # ===========================================

        results[i] = onp.array([batch_size, seq_length/1000., len(dataloader), toc-tic])
        tqdm.write(onp.array_str(results[i][:3], precision=1) + f'{toc-tic:.3f}')

    fpath = os.path.join(args.log_dir, 'profile_'+session_name+'.npy')
    with open(fpath, 'wb') as f:    
        onp.save(f, results)
    print(f"Saved data to: {fpath}")

    return
        

if __name__ == '__main__':
    main()