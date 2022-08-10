# Helper classes for managing killifish data

import os
import h5py

import numpy as onp
import bisect
from functools import partial

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from torch import Generator

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
    
    def argseqsubsets(self, seq_length: int, step_size: int=1,
                      drop_incomplete_seqs: bool=True):
        """Return the list of index tuples to subset data sequentially for given
        sequence length and step size.
        
        If drop_incomplete_seqs is True, resulting sequences will all have
        the same length (seq_length, ). Else, some sequences may be shorter.
        """

        abs_seq_length = seq_length * step_size
        
        subset_keys = []
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

            subset_keys.extend([
                (int(start_idx), int(end_idx), int(step_size))
                for start_idx, end_idx in zip(seq_start_indices, seq_end_indices)
            ])

        return subset_keys

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
        batch_size (int): Number of samples per batch to load.
        seq_length (int): Number of consecutive frames to load per sequence.
        batch_size (int): Number of "independent" sequences to load per minibatch.
        drop_incomplete_seqs (bool): If True, drops sequences which are shorter
            than seq_length. Similar to `drop_last` argument, but when level in.
        drop_last (bool): If True, drops (last) batch if it is not full
        shuffle (bool): If True, shuffles the batches of sequences at every epoch,
            i.e. after every Stopiteration. Else, batches are loaded in the same
            standard alphanumeric order each epoch.
        seed (int): Seed for generating random numbers in a reproducible manner.
            It is recommended to set a large seed, i.e. a number that has a good
            balance of 0 and 1 bits. If None, use default random state. Only
            relevant if shuffle is True.
    """

    def __init__(self, dataset, seq_length=72000, batch_size=1, collate_fn=None,
                 drop_incomplete_seqs=True, drop_last=True,
                 shuffle=False, seed=None):

        # Define strategy to sample batches of data
        # |- Get Dataset keys that index into files to get sequences of seq_length
        step_size = 1
        subset_indices = dataset.argseqsubsets(seq_length, step_size, drop_incomplete_seqs)

        # |- if shuffle, specify RNG and use a Sampler
        if shuffle:
            torch_rng = Generator()
            if seed: torch_rng.manual_seed(seed)
            seq_sampler = SubsetRandomSampler(subset_indices, torch_rng)
        # |- else, just pass in alphanumerically ordered keys
        else:
            torch_rng = None
            seq_sampler = subset_indices
        
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

def arg_uniform_split(target_set_size, set_sizes, set_ids=None):
    """Returns a function that splits the original set of different sizes into
    sets of uniforms sizes, in order. drop_last behavior automatically enforced.

    Suppose we'd like to create even batches of 3 elements from the following
    unevenly-sized sets:
        {a,b,c,d}, {e}, {f,g,h}, {i,j}, {k,l,m,n}
    Each letter is a distinct elemnt within the set. When for-looping through
    this function returns
        [(i=0,s_[0:3]],
        [(i=0,s_[3:4], (i=1,s_[0:1], (i=2,s_[0:1]],
        [(i=2,s_[1:3], (i=3,s_[0:1]],
        [(i=3,s_[1:2], (i=4,s_[0:2]]
    which would result in the sets:
        {a,b,c}, {d,e,f}, {g,h,i}, {j,k,l}

    Example usage:
        num_batches, split_fn = arg_uniform_split(batches_per_file, batch_size)
        i_file = i_in_file = 0      # File index, batch within file index to current file
        for i in range(num_batches):
            b_args, i_file, i_in_file = split_fn([], i_file, i_in_file)

    parameters:
        target_set_size: int
        set_sizes: sequence of ints indicate original set sizes
        set_ids: sequence of ints. Used when shuffling original set sizes
    
    returns:
        list, length (target_num_sets,) of lists with elements (set_id, (frame_start, frame_end))
    """
    def _format(i_set, i_within_start, i_within_end):
        """Defines how resulting args are presented formatted."""
        # return (i_set, np.s_[int(i_within_start):int(i_within_end)])
        return (int(i_set), (int(i_within_start), int(i_within_end)))

    def _fn(n, i_set, i_within_set, out):
        """Recursively returns the args of the sets and elements within sets
        needed to make an even function set.

        Parameters:
            n: int. Number of elements needed to complete an event set.
            i_set: int. Index of current set
            i_within_set: int. Index of current element within set
            out: Current list of arguments

        Returns: 
            i_set: Updated index of current set
            i_within_set: Updated index of unallocated element within set
            out: Updated list of arguments
        """
        n_set = set_sizes[i_set]                   # Size of this set
        
        # This set can complete the remainder of the uniform set: Take as many
        # elements as needed and return to main loop
        if n <= (n_set - i_within_set):
            out.append(_format(set_ids[i_set], i_within_set, i_within_set+n))
            
            # Increase within-set index. If we reached end of set, reset 
            # within-set index to 0 (%) and increase set index
            i_within_set = (i_within_set + n) % n_set
            i_set = i_set + 1 if i_within_set == 0 else i_set
            return i_set, i_within_set, out

        # This set cannot complete the remainder of the uniform set: Take all
        # remaining elements from this set and update number of elements still
        # needed. Then, call function to perform on next set
        else:
            out.append(_format(set_ids[i_set], i_within_set, n_set))
            n -= (n_set - i_within_set)
            return _fn(n, i_set+1, 0, out)

    target_num_sets = sum(set_sizes) // target_set_size                # The number of evenly-sized sets that will result
    set_ids = onp.arange(len(set_sizes)) if set_ids is None else set_ids    
    i_set = i_within_set = 0
    all_args = []
    for _ in range(target_num_sets):
        i_set, i_within_set, set_args = _fn(target_set_size, i_set, i_within_set, [],)
        all_args.append(set_args)
    return all_args

# ==============================================================================

def save_hmm(fpath: str, hmm, **kwargs):
    """Save GaussianHMM in .npz format.
    
    Parameters:
        fpath: str
            Path to file (including name of file)
        hmm: GaussianHMM
        kwargs: Other arrays to save in the same file.
    """
    onp.savez_compressed(fpath, 
        initial_probabilities = hmm.initial_probabilities,
        transition_matrix = hmm.transition_matrix,
        emission_means = hmm.emission_means,
        emission_covariance_matrices = hmm.emission_covariance_matrices,
        **kwargs)
    return 

def load_hmm(fpath: str):
    """Loads a GaussianHMM from .npz format."""
    from ssm_jax.hmm.models import GaussianHMM

    keys = ['initial_probabilities',
            'transition_matrix',
            'emission_means',
            'emission_covariance_matrices']

    with onp.load(fpath) as f:
        hmm = GaussianHMM(**{k: f[k] for k in keys})

    return hmm