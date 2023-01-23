"""Classes for defining datasets and loading data."""
from pathlib import Path
import h5py

import bisect
import itertools
import jax.numpy as jnp
import jax.random as jr
from chex import dataclass

import numpy as onp
from torch.utils.data import Dataset, ConcatDataset, Sampler

# Typing
from chex import Scalar, Array, ArrayTree, PRNGKey, Scalar, Shape
from typing import ( 
    Callable,   # class with __call__() method
    Generic,
    Iterable,   # class with __iter__() method
    Iterator,   # class with __iter__() and __next__() methods
    List,
    Optional,
    Sequence,
    Sized,      # class with __len__() method
    Tuple,
    TypeVar,
    Union
)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
D = TypeVar('D', bound=Optional[ArrayTree])
E = TypeVar('E', bound=ArrayTree)

Pathlike = TypeVar('Pathlike', Path, str)
Slice = TypeVar('Slice') # python slice-like object

# ============================================================================

__all__ = [
    'filter_min_frames',
    'SingleSessionDataset',
    'MultiSessionDataset',
    'SubDataset',
    'random_split'
    'SessionDataset',
    'MultisessionDataset',
]

def filter_min_frames(filepaths: Sequence[Pathlike],
                      min_frames_per_file: int) -> Tuple[List[Pathlike], List[Tuple]]:
    """Filters files for minimum number of frames."""

    # Get the (num_frames, data_dim) for each file
    raw_shapes = [
        h5py.File(fp, 'r')['stage/block0_values'].shape for fp in filepaths
    ]

    # Remove filepaths from dataclass if they do not meet threshold
    has_min_frames = onp.asarray([ds[0] >= min_frames_per_file for ds in raw_shapes])
    if not all(has_min_frames):
        print(f'The following files will be omitted because they have <{min_frames_per_file} frames:')
        for fp in itertools.compress(filepaths, 1-has_min_frames):
            print(f'\t{fp}')
        print()

        filepaths = list(itertools.compress(filepaths, has_min_frames))
        raw_shapes = list(itertools.compress(raw_shapes, has_min_frames))

    return filepaths, raw_shapes

def chunk_into_sequences(key: PRNGKey,
                         num_frames: tuple,
                         length: int,
                         step_size: int) -> List[Slice]:
    """Chunk range into sequences, parameterized by slices."

    Returns
        sequence_slices: sequence of (start, stop, step_size) slices
    """
    # Calculate actual sequence length
    abs_length = length * step_size

    # Generate a random start index
    max_start = num_frames % length
    start = jr.randint(key, shape=(), minval=0, maxval=max_start)

    return [slice(s, int(s+abs_length), int(step_size))
            for s in range(int(start), num_frames-abs_length, abs_length)]


class SingleSessionDataset(Dataset):
    """Dataset class mapping sequence chunks to specific H5 file.
    
    The Fish PC H5 file has the following Pandas-based structure,
    - stage
    | - axis0: column headers (str)
    | - axis1: row indices (int)
    | - *_items, *_values: column headers and data array, blocked by data type
    | - > block0_values: pc data array, f32(num_frames, dim)
    | - > block1_values: frame timestamp, f64(num_frames,)
    | - > block2_values: frame count, uint64(num_frames,)
    
    Since our files are unlikely to be cleanly divisible by the specified
    sequence length, randomly select start index such that we are less
    biased in sampling start vs. end of the files (across multiple files).

    Args:
        filepath: H5 filepath to load
        raw_data_shape: Raw shape of data array from `get_file_shapes`.
        seq_start_key: RNG key ti sgyffke the starting index of sequencing
        seq_length: Sequence length of resulting chunks
        seq_step_size: Step size between frames in resulting chunks
        dtype: dtype of array (float32 vs. float64), much match jax precision
    """

    filepath: Path
    sequence_slices: List[Slice]    # List of slices of valid sequences
    sequence_shape: Shape           # Shape of single sequence of samples
    num_samples: int                # Total number of valid samples
    raw_shape: Shape                # Shape of original data

    def __init__(self,
                 filepath: Pathlike,
                 seq_start_key: PRNGKey,
                 seq_length: int=72_000,
                 seq_step_size: int=1,
                 dtype: str='float32'
                 ):

        self.filepath = Path(filepath)

        with h5py.File(filepath, 'r') as f:
            self.raw_shape = f['stage/block0_values'].shape

        # Get sequence slice indices that define the chunks of this dataset
        # Each file starts at a random index, so that we are less biased in
        # sampling start vs. end of files
        self.sequence_slices = chunk_into_sequences(
            seq_start_key, self.raw_shape[0], seq_length, seq_step_size)

        self.num_samples = len(self.sequence_slices) * seq_length

        self.sequence_shape = (seq_length, *self.raw_shape[1:])

        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.sequence_slices)

    def __getitem__(self, index: int) -> Array:
        """Returns a sequence from file of shape (seq_length, data_dim)"""

        slc = self.sequence_slices[index]
        with h5py.File(self.filepath, 'r') as f:
            data = onp.asarray(f['stage/block0_values'][slc], dtype=self.dtype)

        return data.squeeze()

    def __getitems__(self, indices: List[int]) -> Array:
        """Return set of sequences from single file in a more efficient manner"""

        # Preallocate array
        data = onp.empty((len(indices), *self.sequence_shape), dtype=self.dtype)

        # Fill array
        with h5py.File(self.filepath, 'r') as f:
            for i, index in enumerate(indices):
                slc = self.sequence_slices[index]
                data[i] = onp.asarray(f['stage/block0_values'][slc], dtype=self.dtype)

        return data

class MultiSessionDataset(ConcatDataset):
    """Dataset class mapping sequence chunks to multiple H5 file.

    Since our files are unlikely to be cleanly divisible by the specified
    sequence length, randomly select start index such that we are less
    biased in sampling start vs. end of the files (across multiple files).

    Args:
        filepath: H5 filepath to load
        raw_data_shape: Raw shape of data array from `get_file_shapes`.
        seq_start_key: RNG key ti sgyffke the starting index of sequencing
        seq_length: Sequence length of resulting chunks
        seq_step_size: Step size between frames in resulting chunks
        dtype: dtype of array (float32 vs. float64), much match jax precision
    """
    
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    def __init__(self,
                 filepaths: List[Pathlike],
                 seq_start_key: PRNGKey,
                 seq_length: int=72_000,
                 seq_step_size: int=1,
                 dtype: str='float32'
                 ):

        datasets = [
            SingleSessionDataset(fp, key, seq_length, seq_step_size, dtype)
            for fp, key in zip(filepaths, jr.split(seq_start_key, len(filepaths)))
        ]

        super(MultiSessionDataset, self).__init__(datasets)

    @property
    def sequence_slices(self) -> List[Slice]:
        return [ds.sequence_slices for ds in self.datasets]
    
    @property
    def sequence_shape(self) -> Shape:
        return self.datasets[0].sequence_shape

    @property
    def num_samples(self) -> int:
        return sum([ds.num_samples for ds in self.datasets])

class RandomBatchSampler(Sampler[int]):
    """Custom random batch sampler that can be resumed exactly."""

    data_source: Sized
    batch_size: int
    key: PRNGKey        # Key unique to each epoch
    index: int          # Index within an epoch

    def __init__(self,
                 data_source: Sized,
                 batch_size: int,
                 key: Optional[PRNGKey]=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.key = key
        self.index = 0

    def __len__(self):
        return len(self.data_source) // self.batch_size
    
    def __iter__(self):
        # New epoch
        if self.index >= len(self):
            self.key = jr.split(self.key)[-1]
            self.index = 0

        # Generate shuffled indices (automatically drops last)
        sample_indices = jr.permutation(self.key, len(self.data_source))
        sample_indices = sample_indices[:len(self)*self.batch_size].reshape(len(self), self.batch_size)
    
        while self.index < len(self):
            self.index += 1
            yield sample_indices[self.index-1]
    
    @property
    def state(self):
        return {'key': self.key, 'index': self.index}

    @state.setter
    def state(self, _state):
        self.key = _state['key']
        self.index = _state['index']

# class SubDataset(Dataset):
#     """Subset of a dataset, at the specified indices.
    
#     Args:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#     dataset: Dataset
#     indices: Sequence[int]

#     def __init__(self, dataset: Dataset, indices: Sequence[int]):
#         self.dataset = dataset
#         self.indices = indices

#     def __getitem__(self, idx):
#         if isinstance(idx, (list, tuple)):
#             return self.dataset[[self.indices[i] for i in idx]]
#         return self.dataset[self.indices[idx]]

#     def __len__(self):
#         return len(self.indices)

# def random_split(key: PRNGKey,
#                  dataset: Dataset,
#                  sizes: Sequence[Union[int, float]]) -> List[SubDataset]:
#     """Split dataset into non-overlapping new datasets of specified sizes.
    
#     If a list of fractions that sum up to 1 is given, then the lengths will be
#     computed automatically and 1 count will be distributed in round-robin fashion.
#     """
    
#     # Fractional sizes of dataset are given; convert into number of samples (integer)
#     if all([(sz <= 1 for sz in sizes)]):
#         subset_sizes = [int(frac*len(dataset)) for frac in sizes]

#         # Input sizes sum up to 1, so make sure to distribute all samples
#         if jnp.isclose(sum(sizes), 1) and (sum(subset_sizes) < len(dataset)):
#             remainder = len(dataset) - sum(subset_sizes)
#             for i in range(remainder):
#                 subset_sizes[i % len(sizes)] += 1
        
#         sizes = subset_sizes
    
#     indices = jr.permutation(key, len(dataset))[:sum(sizes)]

#     return [SubDataset(dataset, indices[offset-size:offset])
#             for offset, size in zip(jnp.cumsum(jnp.asarray(sizes)), sizes)]