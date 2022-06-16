# Helper classes for managing killifish data

from os import listdir
from os.path import join, split, isfile

import chex
import warnings

import h5py
import jax.numpy as np
import jax.random as jr
from jax import lax

class FishPCDataset():
    """Dataset of top 15 PCs of a single fish.
    
    TODO Add `load_datetime: bool=False` variable. If False (default), return
    array of PCs. If True, return tuple of arrays (datetimestamp, pcs)

    NB: The HDF5 files are saved from pandas, so it may be natural to reopen
    them through Pandas, but it is incredible faster to use h5py directly.
    For example, the `self._num_frames` setter code takes
        - h5py.File(...)        37.6 ms   +/- 8.0 ms
        - pandas.read_hdf(...)  1min 43 s +/- 24.7 s
    
    Parameters:
        name: str
            Name of fish-specific directory
        data_dir: str
            Path to directory contain data
        data_subset: See options below
            - Load all data: 'all'
            - Load range of data: (i_start, i_end)
            - Load specified files: [fname_0, fname_1, ...]
        min_num_frames: int, default 1.7M.
            If a recording day has fewer than this many frames, omit. Default
            value derived from ~20 Hz x 60 s/min x 60 min/hr x 24 hr/day
            NB: file removal is performed AFTER subsetting into directory, so
            if, for example, data_subset=(i_start, i_end), len(self) may be
            less than (i_end - i_start - 1).
    """

    def __init__(self, name: str, data_dir: str,
                 data_subset='all', min_num_frames: int=1700000
                 ):
        self.name = name

        self._dir = join(data_dir, name)
        self.filenames = sorted([
            f for f in listdir(self._dir) if isfile(join(self._dir, f))
        ])

        if data_subset=='all':
            pass
        elif isinstance(data_subset, tuple) \
             and isinstance(data_subset[0], int) \
             and len(data_subset)==2:
            self.filenames = self.filenames[data_subset[0]:data_subset[1]]
        elif isinstance(data_subset, (tuple, list)) and isinstance(data_subset[0], str):
            i_files = []
            for fname in data_subset:
                try: i_files.append(self.filenames.index(fname))
                except ValueError: print(f"{fname} not found. Continuing.")
            i_files = sorted(i_files)
            self.filenames = [self.filenames[i] for i in i_files]
        else:
            raise ValueError(f"Invalid parameter data_subset={data_subset}.")
        
        # Get data dimensions
        self._dim = \
            h5py.File(join(self._dir, self.filenames[0]), 'r')['stage/block0_values'].shape[-1]
        self._num_frames = np.array([
           len(h5py.File(join(self._dir, f), 'r')['stage/axis1'])
           for f in self.filenames
        ])
        
        # Remove recording days with not enough frames
        self.min_num_frames = min_num_frames
        if min_num_frames:
            i_keep = (self._num_frames >= min_num_frames)
            self.filenames = [
                self.filenames[i] for i, b in enumerate(i_keep) if b
            ]
            self._num_frames = self._num_frames[i_keep]
        return
    
    @property
    def num_frames(self,):
        """Number of frames for each recorded day."""
        return self._num_frames
    
    @property
    def dim(self,):
        """Dimension of observations."""
        return self._dim

    def __len__(self):
        """Total number of recorded days."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Return the i'th recorded day of data, shape (num_frames[index], 15).

        NB: Some days of data are missing due to system crash. These are ignored
        here. Thus, FishDataset[100] may not be the 100th day of recording, but
        may in fact be some later day, e.g. the 105th day of recording if 5 days
        of data are missing.
        """
        with h5py.File(join(self._dir, self.filenames[index]), 'r') as tmp:
            return np.array(tmp['stage/block0_values'][()])
    
    def __add__(self, other):
        raise NotImplementedError

    def train_test_split(self,
                         num_train: int=None,
                         num_test: int=None,
                         frac_train: float=None,
                         frac_test: float=None,
                         seed: jr.PRNGKey=None):
        """Split current dataset into a training set and a testing set.

        One of num_* or frac_* must be specified, for both train and test sizes.
        If both are specified, num_* is retained (and frac_* is ignored).

        If seed is not None, dataset is permuted randomly. Otherwise (default),
        the first N_train + N_test days are used, in sequential order.

        Returns
            train_dataset: FishPCDataset
            test_dataset: FishPCDataset
        """

        assert (num_train is not None) or (frac_train is not None), \
            "One of `num_train[int]` or `frac_train[float]` must be specified."

        assert (num_test is not None) or (frac_test is not None), \
            "One of `num_test[int]` or `frac_test[float]` must be specified."

        num_train = num_train if num_train else int(frac_train * len(self))
        num_test = num_test if num_test else int(frac_test * len(self))

        i_full = jr.permutation(seed, np.arange(len(self))) \
                 if seed is not None else np.arange(len(self))

        # Split dataset by filenames
        f_train = [self.filenames[i] for i in i_full[0:num_train]]
        f_test  = [self.filenames[i] for i in i_full[num_train:num_train+num_test]]

        # Create two new datasets
        _data_dir = split(self._dir)[0]
        train_dataset = FishPCDataset(self.name, _data_dir,
                                      data_subset=f_train,
                                      min_num_frames=self.min_num_frames)
        test_dataset  = FishPCDataset(self.name, _data_dir,
                                      data_subset=f_test,
                                      min_num_frames=self.min_num_frames)
        return train_dataset, test_dataset


class FishPCDataloader():
    """Provides an iterable over the given dataset. Iterator returns an array of
    shape (batch_size, num_frames_per_day, dim). Automatically enforces
    drop_last=True behavior.
    
    Parameters
        dataset: FishPCDataset
        batch_size: int, default 1
            Number of batches to return. If batch_size=-1, load entire dataset.
        num_frames_per_day: int, default -1
            If -1, use MIN_NUM_FRAMES, which is defined by the recording day in
            the dataset with the fewest number of frames. Users may also specify
            a value <= MIN_NUM_FRAMES. This guarantees uniform batch shape. 
        shuffle: bool, default: False
            If True, reshuffle data at every epoch. This is kept track of in
            by the internal `_shuffle_count` attribute.
        speckle: bool, default: False
            If True, uses a sorted, randomly selected set of frames within each
            day of data. If False (default), selects the first num_frames_per_day
            frames to use. Recall that not all frames in a day are used, for
            uniform batch shape purposes. 
        seed: PRNGKey, default: None
            If `shuffle` or `speckle` is True, a global key must be provided.
    """

    def __init__(self,
                 dataset,
                 batch_size:int=1,
                 num_frames_per_day: int=-1,
                 shuffle: bool=False,
                 speckle: bool=False,
                 seed:jr.PRNGKey=None,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size > 0 else len(dataset)

        # By default, self.num_frames_per_day is set to the minimum common number
        # of frames (MIN_NUM_FRAMES). If user passes in specification, ensure
        # that num_frames_per_day <= MIN_NUM_FRAMES to ensure uniform batch size.
        # See how this variable is used in `collate` function...
        self.num_frames_per_day = int(np.min(dataset.num_frames))
        if num_frames_per_day > 0:
            self.num_frames_per_day = \
                    int(np.minimum(num_frames_per_day, self.num_frames_per_day))

        self._batch_shape = (self.batch_size, self.num_frames_per_day, self.dataset.dim)

        # Parameters for randomized iterator
        self.shuffle = shuffle
        self.speckle = speckle
        if shuffle or speckle:
            self._shuffle_key, self._speckle_key = jr.split(seed)
        self._shuffle_count = 0                                                 # "Global counter" to ensure successive calls to this instance results in different randomizations
        return

    @property
    def batch_shape(self) -> chex.Shape:
        """Return shape of each batch of data"""
        return self._batch_shape

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        # Reset this iterator instance's counter
        self._iter_count = 0

        # Create data buffer to reduce memory footprint
        self._buffer = np.empty(self.batch_shape)

        # Create (maybe random) index into dataset, shape (num_batches, batch_size)
        self.idx_into_ds = np.arange(0, len(self.dataset))
        if self.shuffle:
            key = jr.fold_in(self._shuffle_key, self._shuffle_count)
            self.idx_into_ds = jr.permutation(key, len(self.dataset))
            self._shuffle_count += 1
        
        self.idx_into_ds = self.idx_into_ds[:len(self)*self.batch_size]
        self.idx_into_ds = self.idx_into_ds.reshape(-1, self.batch_size)
       
        return self
    
    def __next__(self):
        if self._iter_count >= len(self):
            raise StopIteration

        i_ds_batch = self.idx_into_ds[self._iter_count]                         # shape (batch_size)
        for i, i_ds in enumerate(i_ds_batch):
            if self.speckle:                                                    # This is probably very slow code...
                key = jr.fold_in(self._speckle_key, self._shuffle_count*self._iter_count + i)
                idx_into_day = jr.permutation(key, self.dataset.num_frames[i_ds])
                idx_into_day = np.sort(idx_into_day[:self.num_frames_per_day])
                self._buffer = self._buffer.at[i].set(self.dataset[i_ds][idx_into_day])
            else:
                self._buffer = self._buffer.at[i].set(self.dataset[i_ds][:self.num_frames_per_day])
        
        self._iter_count += 1

        return self._buffer