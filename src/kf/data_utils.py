# Helper classes for managing killifish data

from os import listdir
from os.path import join, split, isfile

import h5py
import numpy as onp

from torch.utils.data as Dataset, DataLoader, SubsetRandomSampler\
from torch import Generator

def _extract_file_metadata(filepath):
    """Extract metadata associated with a given file."""
    
    # fish_id
    # day born, day died, cohort id?
    num_frames = len(h5py.File(f, 'r')['stage/axis1'])

    raise NotImplementedError

class FishPCDataset(Dataset):
    """
    Maps (fish_id, file_id, start_frame_idx, end_frame_idx) to the corresponding
    consecutive data sequence with shape (seq_length, data_dim) and data labels
    () where seq_length <= (end_frame_idx-start_frame_idx).

    The length of Dataset is considered the total number of emissions, across
    all fish and files. Note that this number may be more than the total number
    of emissions that the corresonding DataLoader may load.

    Args:
        filepaths ([str, iterable[str]]): path or list of paths to H5 files

    Returns:

    """

    def __init__(self, filepaths, dtype=onp.float32):

        # Standardize filepaths to be tuple of strings
        if isinstance(filepaths, str):
            self.filepaths = (filepaths,)
        elif isinstance(filepaths, (list, tuple)):
            if all(isinstance(path, str) for path in filepaths):
                self.filepaths = tuple(filepaths)
            else:
                raise ValueError("Expected all elements of `filepaths` to be strings.")
        else:
            raise ValueError(
                "Expected `filepaths` to be string or an iterable of strings, " \
                + f"received {type(filepaths)}.")

        # Extract metadata from each file
        self.metadata = {
            lambda f: _extract_metadata(f) for f in self.filepaths
        }

    @property
    def dim(self,):
        """Return data dimension of dataset."""
        pass 

    def __len__(self,):
        """Return total number of frames, across all fish and files."""
        pass
    
    def __getitem__(self, key):
        """Return specificed data array and identifying labels
        Args:
            key (tuple), consisting of elements
                fish_id (str)
                file_id (str)
                start_frame_idx (int)
                end_frame_idx (int)
        
        Returns:
            data (onp.ndarray), shape (seq_length, obs_dim)
            label (tuple), consisting of elements
                fish_id (str)
                timestamps (onp.ndarray, float64 dtype)
        """
        raise NotImplementedError

        # TODO
        def _get_filepath(fish_id, file_id):
            raise NotImplementedError

        fish_id, file_id, start_frame_idx, end_frame_idx = key

        with h5py.File(_get_filepath(fish_id, file_id)), 'r') as tmp:
            # TODO return data (here), return labels
            return onp.array(tmp['stage/block0_values'][()])

def FishPCLoader(DataLoader):
    """Provides an iterable over FishPCDataset. Randomly selects subsets of
    consecutive frames of length (seq_length,) from the Dataset, returns a batch
    of emissions of shape (batch_size, seq_length, data_dim).

    Args:
        dataset (FishPCDataset): Dataset from which to load data.
        batch_size (int): Number of samples per batch to load.
        seq_length (int): Number of consecutive frames to load per sequence.
        batch_size (int): Number of "independent" sequences to load per minibatch.
        drop_incomplete_batches (bool): If True, drops batches which are smaller
            than seq_length. Similar to `drop_last` argument, but handled in a 
            custom fashion, since each by data file may have an incomplete batch.
        shuffle (bool): If True, shuffles the batches of sequences at every epoch,
            i.e. after every Stopiteration. Else, batches are loaded in the same
            standard alphanumeric order each epoch.
        seed (int): Seed for generating random numbers in a reproducible manner.
            It is recommended to set a large seed, i.e. a number that has a good
            balance of 0 and 1 bits. If None, use default random state. Only
            relevant if shuffle is True.
    """

    def __init__(self, dataset, seq_length=7200, batch_size=1,
                 drop_incomplete_batches=True, shuffle=False, seed=None):

        # Define strategy to sample batches of data
        # |- Get Dataset keys that index into files to get sequences of seq_length
        subset_keys = self._make_subset_keys(dataset, seq_length, drop_incomplete_batches)

        # |- if shuffle, specify RNG and use a Sampler
        if shuffle:
            torch_rng = Generator()
            if seed: torch_rng.manual_seed(seed)
            seq_sampler = SubsetRandomSampler(subset_keys, torch_rng)
        # |- else, just pass in alphanumerically ordered keys
        else:
            torch_rng = None
            seq_sampler = subset_keys
        
        batch_sampler = BatchSampler(seq_sampler, batch_size, drop_last)

        # Function to merge a list of sequences to form a mini-batch
        collate_fn = lambda batch: jnp.asarray(onp.stack(batch, axis=0))

        super.(self.__class__, self).__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            generator=torch_rng,
        )

    @static_method
    def _make_subset_keys(dataset, seq_length: int, drop_incomplete_batches: bool):
        """Construct a list of keys to take subsets of data from the dataset.
        
        If drop_incomplete_batches is True, resulting sequences will all have
        the same length (seq_length, ). Else, some sequences may be shorter.
        """

        subset_keys = []
        for _metadata in dataset.file_metadata:
            fish_id = _metadata['fish_id']
            file_id = _metadata['file_id']
            num_frames = _metadata['num_frames']
            start_indices = onp.arange(0, num_frames, seq_length)
            
            if drop_incomplete_batches:
                end_indices = start_indices + seq_length
                if end_indices[-1] >= num_frames:
                    start_indices = start_indices[:-1]
                    end_indices = end_indices[:-1]
            else:
                end_indices = onp.minimum(start_indices+seq_length, num_frames)

            subset_keys.extend([
                (fish_id, file_path, start_idx, end_idx)
                for start_idx, end_idx in zip(start_indices, end_indices)
            ])

        return subset_keys

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
    set_ids = np.arange(len(set_sizes)) if set_ids is None else set_ids    
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
    import numpy as np  # jax.numpy has not yet implemented this method
    np.savez_compressed(fpath, 
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

    with np.load(fpath) as f:
        hmm = GaussianHMM(**{k: f[k] for k in keys})

    return hmm