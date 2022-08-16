# Helper classes for managing killifish data

import os
import glob
import h5py

import numpy as onp
import jax.random as jr
import bisect
from functools import partial

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from torch import Generator

from ssm_jax.hmm.models import GaussianHMM

__all__ = [
    'FishPCDataset',
    'FishPCLoader',
    'CheckpointDataclass',
    'train_and_checkpoint',
]

class FishPCDataset(Dataset):
    """
    Dataset class wrapping PC data of a single fish across its whole lifespan.
    Maps (file_id, start_frame_idx, end_frame_idx) to the corresponding
    consecutive data sequence with shape (seq_length, data_dim) and data labels
    () where seq_length <= (end_frame_idx-start_frame_idx).

    The length of Dataset is considered the total number of emissions, across
    all fish and files. Note that this number may be more than the total number
    of emissions that the corresonding DataLoader may load.

    Each H5 file has the following Pandas-based structure,
    - stage
    | - axis0: column headers (str)
    | - axis1: row indices (int)
    | - *_items, *_values: column headers and data array, blocked by data type
    | - > block0_values: pc data array, f32(num_frames, dim)
    | - > block1_values: frame timestamp, f64(num_frames,)
    | - > block2_values: frame count, uint64(num_frames,)
    
    Args:
        filepaths ([str, iterable[str]]): path or list of paths to H5 files.
        **metadata (dict):
    Returns:

    """

    def __init__(self, filepaths, **metadata):
        # Standardize filepaths to be tuple of strings; sort alphabetically.
        if isinstance(filepaths, str):
            filepaths =(filepaths,)
        elif not isinstance(filepaths, (list, tuple)):
            raise ValueError(
                "Expected `filepaths` to be string or an iterable of strings, " \
                + f"received {type(filepaths)}.")
        
        self._filepaths = sorted([
            fp for fp in filepaths if os.path.isfile(fp) and fp.endswith('.h5')
        ],)
        assert len(self._filepaths) > 0, 'No files found. Make sure that e.g. absolute paths are passed in.'

        # Store data (obs) dimensions, cumulative sum of number of frames per file
        file_data_shape = onp.stack([
            h5py.File(fp, 'r')['stage/block0_values'].shape for fp in self._filepaths
        ],)
        self._num_frames_per_file = file_data_shape[:,0]
        self._cumulative_frames = onp.cumsum(self._num_frames_per_file)
        self._dim = file_data_shape[-1,-1]
              
        # TODO Pass in metadata and process (if needed)
        self.metadata = metadata

    @property
    def dim(self,):
        """Return data dimension of dataset."""
        return self._dim
    
    @property
    def cumulative_frames(self,):
        """Return data dimension of dataset."""
        return self._cumulative_frames

    def __len__(self,):
        """Return total number of frames, across all fish and files."""
        return self._cumulative_frames[-1]
    
    def slice_seq(self, seq_length, step_size=1, drop_incomplete_seqs=True):
        """Return the list of index tuples to slice into data sequentially for
        given sequence length and step size.
        Each index tuple produces values in the interval [start, stop) where
            seq_length = (stop - start) * step_size.
        
        Args:
            seq_length (int): Number of consecutive frames in a sequence
            step_size (int): Spacing between consecutive frames
            drop_incomplete_seqs (bool): If True, slices would result in sequences
                with the same length (seq_length, ). If False, some sequences
                may be shorter. Default: True
        Returns:
            slices (list): List of tuples consist of (start, stop, step) values
        """

        abs_seq_length = seq_length * step_size
        
        slices = []
        for i_file in range(len(self.cumulative_frames)):
            f_start_idx = 0 if i_file == 0 else self.cumulative_frames[i_file-1] 
            f_end_idx = self.cumulative_frames[i_file] 
            seq_start_indices = onp.arange(f_start_idx, f_end_idx, abs_seq_length)
            
            if drop_incomplete_seqs:
                seq_end_indices = seq_start_indices + abs_seq_length
                if seq_end_indices[-1] >= f_end_idx:
                    seq_start_indices = seq_start_indices[:-1]
                    seq_end_indices = seq_end_indices[:-1]
            else:
                seq_end_indices = \
                    onp.minimum(seq_start_indices + abs_seq_length, f_end_idx)

            slices.extend([
                (int(start_idx), int(end_idx), int(step_size))
                for start_idx, end_idx in zip(seq_start_indices, seq_end_indices)
            ])

        return slices

    def split_seq(self, split_sizes, seed=None,
                  seq_length=72000, step_size=1, drop_incomplete_seqs=True):
        """Return list of index tuples that split dataset into specified sizes.

        Parameters:
            split_sizes (list or tuple, length S): Float or int of number of
                sequences per split. If float, interpreted as percentage of all
                slices. Else, if int, interpreted as number of slices.
            seed (jr.PRNGKey or None): If specified, sequence slices are shuffled.
                Else, if None, sequence slices are kept sequentially.
            seq_length, step_size, drop_incomplete_seqs: See `get_slices` fxn.
        
        Returns:
            split_slices (array-like, length S): List of array-like of tuples.
        """

        all_slices = self.slice_seq(seq_length, step_size, drop_incomplete_seqs)
        n_slices = len(all_slices)        
        
        idx = onp.asarray(jr.permutation(seed, n_slices)) \
              if seed is not None else onp.arange(n_slices)
        
        # Standardize split sizes into number of sequence slices.
        split_sizes = [
            int(size*n_slices) if size < 1 else int(size)
            for size in split_sizes
        ]
        
        # Start and stop indices of each split
        split_indices = onp.concatenate([onp.zeros(1), onp.cumsum(split_sizes)])
        split_indices[-1] = onp.minimum(split_indices[-1], n_slices)
        split_indices = split_indices.astype(int)

        return [
            tuple(onp.asarray(all_slices)[idx[start:stop]])
            for start, stop in zip(split_indices[:-1], split_indices[1:])
        ]

    def __getitem__(self, ds_indices):
        """Return specificed data array and identifying labels.

        Only returns data from a single file, i.e. the file indexed by start_idx.

        Args:
            indices (int or tuple[int]), consisting of
                (ds_start_idx, [ds_end_idx, [step_size]]).
        
        Returns:
            data (onp.ndarray), shape (seq_len, obs_dim) where seq_len is
                (ds_end_idx - ds_start_idx) // step_size. Array is 2d
            label (tuple), consisting of elements
                timestamps (onp.ndarray, shape (seq_len,))
        """

        try: # int-like
            ds_indices = (int(ds_indices), int(ds_indices)+1, 1)
        except TypeError: # ds_indices is not an int or scalar
            pass
        try: # array-like
            if len(ds_indices) == 1:
                ds_indices = (int(ds_indices), int(ds_indices)+1, 1)
            elif len(ds_indices) == 2:
                ds_indices = (int(ds_indices[0]), int(ds_indices[1]), 1)
            elif len(ds_indices) == 3:
                ds_indices = (int(ds_indices[0]), int(ds_indices[1]), int(ds_indices[2]))
            else:
                raise ValueError(f'Expected int or array-like with length <= 3, received {len(ds_indices)}')
        except TypeError: # ds_indices is also not array-like
            raise TypeError(f'Expected int or array-like with length <= 3, received {type(ds_indices)}')
        
        ds_start_idx, ds_end_idx, step_size = ds_indices
        abs_seq_length = ds_end_idx - ds_start_idx
        
        i_file = bisect.bisect_right(self.cumulative_frames, ds_start_idx)
        f_start_idx = 0 if i_file == 0 else self.cumulative_frames[i_file-1] 
        f_end_idx = self.cumulative_frames[i_file] 
        
        start_idx = ds_start_idx - f_start_idx
        end_idx = onp.minimum(start_idx + abs_seq_length, f_end_idx)
        with h5py.File(self._filepaths[i_file], 'r') as f:
            data = onp.asarray(
                f['stage/block0_values'][start_idx:end_idx:step_size],
                dtype=onp.dtype('float32')
            )

            timestamp = onp.asarray(
                f['stage/block1_values'][start_idx:end_idx:step_size],
                dtype=onp.dtype('float64')
            )

        # TODO Return (fish_id, age, time-of-day) instead of timestamp
        return data.squeeze(), timestamp.squeeze()

    @staticmethod
    def collate(sequences):
        """Stack a list of sequences (or samples) along new leading dimension.
        This function will be dependent on what __getitem__ returns.
        """
        # If uniform sequence lengths, return tuple of stacked arrays
        try:
            return tuple(map(onp.stack, zip(*sequences)))
        # If non-uniform sequence lengths, return tuple of lists
        except ValueError:
            return tuple(zip(*sequences))
        
class FishPCLoader(DataLoader):
    """Provides an iterable over FishPCDataset. Randomly selects subsets of
    consecutive frames of length (seq_length,) from the Dataset, returns a batch
    of emissions of shape (batch_size, seq_length, data_dim).

    Args:
        dataset (FishPCDataset): Dataset from which to load data.
        seq_slices (list): Specifies the subset (or whole set) of the dataset to
            load. List of (start, stop, step) slice tuples. Generate via
            FishPCDataset.get_slices(...) or FishPCDataset.split(...)
        batch_size (int): Number of "independent" sequences to load per minibatch.
        drop_last (bool): If True, drops (last) batch if it is not full
        shuffle (bool): If True, shuffles the batches of sequences at every epoch,
            i.e. after every Stopiteration. Else, batches are loaded in the same
            standard alphanumeric order each epoch.
        seed (int): Seed for generating random numbers in a reproducible manner.
            It is recommended to set a large seed, i.e. a number that has a good
            balance of 0 and 1 bits. If None, use default random state. Only
            relevant if shuffle is True.
    """

    def __init__(self, dataset, seq_slices, batch_size=1, collate_fn=None,
                 drop_last=True, shuffle=False, seed=None):

        self.total_emissions = sum(
            [(stop-start)//step for start, stop, step in seq_slices]
        )

        # Define strategy to sample batches of data
        # |- if shuffle, specify RNG and use a Sampler
        if shuffle:
            torch_rng = Generator()
            if seed: torch_rng.manual_seed(seed)
            seq_sampler = SubsetRandomSampler(seq_slices, torch_rng)
        # |- else, just pass in alphanumerically ordered keys
        else:
            torch_rng = None
            seq_sampler = seq_slices
        
        batch_sampler = BatchSampler(seq_sampler, batch_size, drop_last)

        # Function to merge a list of sequences to form a mini-batch
        collate_fn = dataset.collate if collate_fn is None else collate_fn

        super(self.__class__, self).__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            generator=torch_rng,
        )
    
# ==============================================================================

@dataclass
class CheckpointDataclass:
    """Container for holding training checkpoint information.
    
    Args
        directory (str): Path to directory to store checkpoint files in.
        prefix (str): Prefix of checkpoint file names.
        interval (int): Number of epochs to train between checkpoints.
            Default: -1 (no intermediate checkpointing).
        keep (int): Number of past checkpoints to keep. Default: -1 (keep all).
            TODO: Implement in save_checkpoint (pop first and add to last);
            Implement in post_init, to find all existing checkpoints
    """
    directory: str
    prefix: str='ckp'
    interval: int=-1
    keep: int=-1

    def _make_file_path(self, val):
        if isinstance(val, int):
            return os.path.join(self.directory, f'{self.prefix}_{val:03d}.ckp.npz')
        else:
            return os.path.join(self.directory, f'{self.prefix}_{val}.ckp.npz')

    def _get_existing_files(self,):
        return glob.glob(self._make_file_path('*'))

    def save(self, hmm, this_epoch, **arrs):
        """
            hmm (GaussianHMM): Model to save
            this_epoch (int): Epoch this HMM just finished training.
            **arrs: dict of additional arrays to save
        """
        ckp_path = self._make_file_path(this_epoch+1-self.interval)

        # Make directory (recursively) if it does not exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # TODO Is there a better way to save using i.e. hmm.unconstrained_params?
        onp.savez_compressed(
            ckp_path,
            initial_probabilities = hmm.initial_probs.value,
            transition_matrix = hmm.transition_matrix.value,
            emission_means = hmm.emission_means.value,
            emission_covariance_matrices = hmm.emission_covariance_matrices.value,
            epochs_completed = this_epoch,
            **arrs)
        
        return ckp_path
    
    def load(self, path):
        """Load checkpoint from specified path."""

        hmm_keys = ['initial_probabilities',
                    'transition_matrix',
                    'emission_means',
                    'emission_covariance_matrices']
        
        with onp.load(path) as f:
            epochs_completed = f['epochs_completed']
            hmm = GaussianHMM(**{k: f[k] for k in hmm_keys})
            all_lps = f['all_lps'] if 'all_lps' in f else []
            
        return hmm, epochs_completed, all_lps
    
    def load_latest(self, return_path=False):
        """Load latest checkpoint (alphanumerically) in this instance's directory.
        
        Returns:
            hmm (GaussianHMM)
            prev_epoch (int):
            prev_lps (array-like):
            ckp_path (str, optional): Path of file, returned if return_path=True
        """

        existing_ckps = sorted(self._get_existing_files())
        
        if existing_ckps:
            last_ckp_path = existing_ckps[-1]
            out = self.load(existing_ckps[-1])
        else:   # No checkpoints in this directory
            last_ckp_path = None
            out = (None, -1, [])
        
        return (*out, last_ckp_path) if return_path else out

def train_and_checkpoint(emissions_loader,
                         hmm: GaussianHMM,
                         num_epochs: int,
                         checkpoint: CheckpointDataclass,
                         starting_epoch: int=0,
                         prev_lps=[],
                         ):
    """Fit HMM via stochastic EM, with intermediate checkpointing.

    After all training, HMM will automatically be checkpointed a final time.

    Args
        emissions_loader (torch.utils.data.DataLoader): Iterable over emissions
        hmm (GaussianHMM): GaussianHMM to train. May be partially trained.
        num_epochs (int): (Total) number of epochs to train.
        checkpoint (CheckpointDataclass): Container with checkpoint parameters.
        starting_epoch (int): Starting epoch of this training (i.e. hmm has already
            been partially trained; warm-start). Default: 0 (cold-start).
        prev_lps (array-like, length starting_epoch): Mean expected log
            probabilities from previous epochs, if HMM is being warm-started.
            Default: [] (cold-start, no previous training).
    
    Returns
        all_lps (array-like, length num_epochs): Mean expected log probabilities
        ckp_path (str): Path to last checkpoint file
    """
    assert starting_epoch >= 0
    assert checkpoint.interval < num_epochs  

    # If no interval specified, run (remaining) number of epochs
    if checkpoint.interval < 1:
        checkpoint.interval = num_epochs - starting_epoch

    all_lps = prev_lps
    for last_epoch in range(starting_epoch, num_epochs, checkpoint.interval):

        lps = hmm.fit_stochastic_em(emissions_loader,
                                    emissions_loader.total_emissions,
                                    num_epochs=checkpoint.interval)
        all_lps = onp.concatenate([all_lps, lps])
        this_epoch = last_epoch + checkpoint.interval
        ckp_path = checkpoint.save(hmm, this_epoch, all_lps=all_lps)

    return all_lps, ckp_path