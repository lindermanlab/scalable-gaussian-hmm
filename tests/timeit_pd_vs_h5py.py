# Compare times for Pandas vs H5py access of .h5 files

import timeit

from os import listdir
from os.path import join, isfile

import pandas as pd
import h5py
import numpy as np

DATAPATH = '/home/libi/killifish/data/fish0_137/'
filenames = sorted([
        f for f in listdir(DATAPATH) if isfile(join(DATAPATH, f))
    ])
index = 42

# =============================================================================

# 61.7 ms +/- 1.87 ms per loop (7 runs, 1 loop each)
def get_num_frames_h5py():
    return [len(h5py.File(join(DATAPATH, f), 'r')['stage']['axis1'])
            for f in filenames]

# 1min 43 s +/- 24.7 s
def get_num_frames_pandas():
    return [len(pd.read_hdf(join(DATAPATH, f)))
            for f in filenames]

# -----------------------------------------------------------------------------

# 30.7 ms +/- 2.92 ms per loop (7 runs, 10 loops each)
def get_data_h5py():
    return h5py.File(join(DATAPATH, filenames[index]), 'r')['stage/block0_values'][()]

# 471 ms +/- 10.1 ms per loop (7 runs, 1 loop each)
def get_data_pandas():
    df = pd.read_hdf(join(DATAPATH, filenames[index]))
    return np.stack([df[f'pc_{i}'] for i in range(15)], axis=-1)