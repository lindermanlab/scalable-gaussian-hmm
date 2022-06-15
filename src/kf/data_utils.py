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
                try:
                    i_files.append(self.filenames.index(fname))
                except ValueError:
                    print(f"{fname} not found. Continuing.")
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

        _data_dir = split(self._dir)[0]
        train_dataset = FishPCDataset(self.name, _data_dir,
                                      data_subset=f_train,
                                      min_num_frames=self.min_num_frames)
        test_dataset  = FishPCDataset(self.name, _data_dir,
                                      data_subset=f_test,
                                      min_num_frames=self.min_num_frames)
        return train_dataset, test_dataset


class FishPCDataloader():
    """Provides an iterable over the given dataset. Iterator returns array of
    shape (batch_size, num_frames_per_day * num_days_per_batch, dim).
    
    Parameters
        dataset: FishPCDataset
        batch_size: int, default 1
            Number of batches to return.
        num_days_per_batch: int, default 1
            Number of consecutive days per batch. TODO: Remove
        num_frames_per_day: int, default -1
            If -1, use all frames.
        uniform_batch_size: bool, default True
            If True, all batches will have the same number of frames. TODO: Remove
        drop_last: bool. default False
            If True, drop the last incomplete batch. If False, keep the last
            batch, which be smaller if dataset is indivisible by batch size,
            If uniform_batch_size=True, then drop_last=True.
        shuffle: bool, default: False
            If True, reshuffle data at every epoch. This is kept track of in
            by the internal `shuffle_count` attribute.
        shuffle_key: PRNGKey, default: None
            If `shuffle` is True, a global key must be provided.
    """

    def __init__(self,
                 dataset,
                 batch_size:int=1,
                 num_days_per_batch:int=1,
                 num_frames_per_day: int=-1,
                 uniform_batch_size:bool=True,
                 drop_last: bool=True,
                 shuffle:bool=False,
                 shuffle_key:jr.PRNGKey=None,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_days_per_batch = num_days_per_batch                            # "batch_size", but renamed for clarity

        warnings.warn('Forcing `uniform_batch_size=True`.')
        uniform_batch_size = True
        # See how this variable is used in `collate` function

        # By default, self.num_frames_per_day is set to the minimum common number
        # of frames. If user passes in option, ensure it is less than this number.
        self.num_frames_per_day = np.min(dataset.num_frames)
        if num_frames_per_day > 0:
            self.num_frames_per_day = np.minimum(num_frames_per_day, self.num_frames_per_day)
        
        self.drop_last = True if uniform_batch_size else drop_last

        self._data_batch_shape = (self.batch_size, self.num_frames_per_day, self.dataset.dim)

        # Parameters for randomized iterator
        self.shuffle = shuffle
        if shuffle and (shuffle_key is None):
            raise ValueError("Since random shuffling indicated, must provide `shuffle_key` PRNGKey.")
        self.shuffle_key = shuffle_key
        self.shuffle_count = 0                                                  # "Global counter"
        return

    @property
    def data_batch_shape(self) -> chex.Shape:
        """Return shape of each batch of data"""
        return self._data_batch_shape

    def __len__(self) -> int:
        effective_batch_size = self.num_days_per_batch * self.batch_size
        # effective_batch_size = self.batch_size
        
        if self.drop_last:
            return len(self.dataset) // effective_batch_size
        else:
            from math import ceil 
            return ceil(len(self.dataset) / effective_batch_size)

    def __iter__(self,):
        self.indices_into_dataset = np.arange(
            0, len(self) * self.num_days_per_batch * self.batch_size,
            self.num_days_per_batch)

        if self.shuffle:
            self.indices_into_dataset = \
                jr.permutation(jr.fold_in(self.shuffle_key, self.shuffle_count),
                               self.indices_into_dataset)                       # If shuffle, randomly permute the indices
            self.shuffle_count += 1                                             # Update global counter
        
        self.indices_into_dataset = \
                        self.indices_into_dataset.reshape(-1, self.batch_size)
        assert len(self) == len(self.indices_into_dataset)

        self.iter_count = 0                                                     # Reset this instance's counter
        return self
    
    def __next__(self,):
        if self.iter_count >= len(self):
            raise StopIteration

        # TODO: Note that if shuffle and if drop_last=False, the small batch
        # will be located somewhere in the iterator and not necessarily at the 
        # end. This may be undesirable behavior. Solutions: a) force parameter
        # drop_last=True if shuffle=True; b) find another way to construct
        # self.indices_into_dataset so that last days are not necessarily ignored
        i_ds_start = self.indices_into_dataset[self.iter_count]
        i_ds_stop = i_ds_start + self.num_days_per_batch

        # If drop_last == False, ensure last batch doesn't raise out of range error
        i_ds_stop = np.where(
            i_ds_stop > len(self.dataset),
            i_ds_stop - (self.num_days_per_batch - len(self.dataset) % self.num_days_per_batch),
            i_ds_stop
            )

        # Update counter
        self.iter_count += 1

        return np.stack([
            self.collate(np.arange(_start, _stop))
            for _start, _stop in zip(i_ds_start, i_ds_stop)
        ], axis=0)

    def collate(self, indices):
        """Batches data along frames axis.

        If uniform_batch_size is True, resulting array has shape
            (len(indices)*num_frames_per_day, emission_dim)
        Else, resulting array has shape
            (sum_i len(ds[i]), emission_dim)
        """
        return np.concatenate([
            self.dataset[i_ds][:self.num_frames_per_day] for i_ds in indices
            ], axis=0)