# Helper classes for managing killifish data

from os import listdir
from os.path import join, isfile
import pandas as pd
import h5py

import jax.numpy as np

class FishPCDataset():
    """Dataset of top 15 PCs of a single fish.
    
    TODO Add `load_datetime: bool=False` variable. If False (default), return
    array of PCs. If True, return tuple of arrays (datetimestamp, pcs)

    NB: The HDF5 files are saved from pandas, so it may be natural to reopen
    them through Pandas, but it is incredible faster to use h5py directly.
    For example, the `self._num_frames` setter code takes
        - h5py.File(...)        37.6 ms   +/- 8.0 ms
        - pandas.read_hdf(...)  1min 43 s +/- 24.7 s
    """

    pc_names = [f'pc_{i}' for i in range(15)]

    def __init__(self, fish_name: str, data_dir: str):
        self.name = fish_name

        self.fish_dir = join(data_dir, fish_name)
        self.file_names = sorted([
            f for f in listdir(self.fish_dir) if isfile(join(self.fish_dir, f))
        ])

        self._num_frames = [
           len(h5py.File(join(self.fish_dir, f), 'r')['stage/axis1'])
            for f in self.file_names
        ]

        return

    @property
    def num_frames(self):
        """Number of frames for each recorded day."""
        return self._num_frames

    def __len__(self):
        """Total number of recorded days."""
        return len(self.file_names)

    def __getitem__(self, index):
        """Return the i'th recorded day of data, shape (num_frames[index], 15).

        NB: Some days of data are missing due to system crash. These are ignored
        here. Thus, FishDataset[100] may not be the 100th day of recording, but
        may in fact be some later day, e.g. the 105th day of recording if 5 days
        of data are missing.
        """
        with h5py.File(join(self.fish_dir, self.file_names[index]), 'r') as tmp:
            return np.array(tmp['stage/block0_values'][()])
    
    def __add__(self, other):
        raise NotImplementedError