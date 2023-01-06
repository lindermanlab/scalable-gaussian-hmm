"""Classes for defining datasets and loading data."""
from pathlib import Path
import h5py

import bisect
import itertools
import jax.numpy as jnp
import jax.random as jr
from chex import dataclass

# Typing
from chex import Scalar, Array, ArrayTree, PRNGKey, Scalar, Shape
from typing import ( 
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
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
    'Dataset',
    'ConcatDataset',
    'SubDataset',
    'random_split'
    'SessionDataset',
    'MultisessionDataset',
]

@dataclass
class Dataset():
    """An abstract class that represents a map from keys to data samples."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> E:
        raise NotImplementedError

    def __getitems__(self, indices) -> List[E]:
        return [self.__getitem__(idx) for idx in indices]

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset([self, other])

class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    Ported from torch.utils.data.ConcateDataset

    Args:
        datasets (sequence): List of datasets to be concatenated.
    """
    
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence: Sequence[T]) -> List[int]:
        """Calculate cumulative sum of objects in a sequence"""
        r, s = [], 0
        for obj in sequence:
            l = len(obj)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]):
        super(ConcatDataset, self).__init__()
        
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

class SubDataset(Dataset):
    """Subset of a dataset, at the specified indices.
    
    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(key: PRNGKey,
                 dataset: Dataset,
                 sizes: Sequence[Union[int, float]]) -> List[SubDataset]:
    """Split dataset into non-overlapping new datasets of specified sizes.
    
    If a list of fractions that sum up to 1 is given, then the lengths will be
    computed automatically and 1 count will be distributed in round-robin fashion.
    """
    
    # Fractional sizes of dataset are given; convert into number of samples (integer)
    if all([(sz <= 1 for sz in sizes)]):
        subset_sizes = [int(frac*len(dataset)) for frac in sizes]

        # Input sizes sum up to 1, so make sure to distribute all samples
        if jnp.isclose(sum(sizes), 1) and (sum(subset_sizes) < len(dataset)):
            remainder = len(dataset) - sum(subset_sizes)
            for i in range(remainder):
                subset_sizes[i % len(sizes)] += 1
        
        sizes = subset_sizes
    
    indices = jr.permutation(key, len(dataset))[:sum(sizes)]

    return [SubDataset(dataset, indices[offset-size:offset])
            for offset, size in zip(jnp.cumsum(jnp.asarray(sizes)), sizes)]

# =============================================================================
# Project-specific dataclasses
# =============================================================================

@dataclass
class IteratorState():
    key: PRNGKey            # Key uniquely defining an epoch
    index: Scalar           # Index within current epoch

def default_collate_fn(data: Sequence) -> Array:
    """Pass back the data as is."""
    return jnp.asarray(data)

def default_sampler_fn(key: PRNGKey, dataset: Dataset) -> List[int]:
    """Sequentially iterate through the dataset."""
    return range(len(dataset))

class RandomBatchDataloader(Iterable):
    dataset: Dataset
    batch_size: int
    sample_fn: Callable[[PRNGKey, Dataset], List[int]]
    collate_fn: Callable[[Sequence], Array]
    key: PRNGKey
    index: int

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 key: Optional[PRNGKey]=None,
                 sample_fn: Optional[Callable[[PRNGKey, Dataset], List[int]]]=None,
                 collate_fn: Optional[Callable[[Sequence], Array]]=None,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_fn = default_sampler_fn if sample_fn is None else sample_fn
        self.collate_fn = default_collate_fn if collate_fn is None else collate_fn

        self.key = key
        self.index = 0

    def __len__(self):
        # number of batches that this iterator will produce
        return len(self.dataset) // self.batch_size

    @property
    def state(self):
        return IteratorState(key=self.key, index=self.index)

    @state.setter
    def state(self, iterator_state):
        self.key = iterator_state.key
        self.index = iterator_state.index

    def __iter__(self):
        # New epoch
        if self.index >= len(self):
            self.key = jr.split(self.key)[-1]
            self.index = 0

        # Generate shuffled indices (automatically drops last)
        sample_indices = self.sample_fn(self.key, self.dataset)
        sample_indices = sample_indices[:len(self)*self.batch_size].reshape(len(self), self.batch_size)
    
        while self.index < len(self):
            self.index += 1

            yield self.collate_fn(
                self.dataset.__getitems__(sample_indices[self.index-1])
            )

# =============================================================================
# Project-specific dataclasses
# =============================================================================

def get_file_raw_shapes(filepaths: Sequence[Pathlike],
                        min_frames_per_file: int) -> Tuple[List[Pathlike], List[Tuple]]:
    """Get number of frames for each file. Remove files without enough frames."""

    # Get the (num_frames, data_dim) for each file
    raw_shapes = [
        h5py.File(fp, 'r')['stage/block0_values'].shape for fp in filepaths
    ]

    # Remove filepaths from dataclass if they do not meet threshold
    has_min_frames = [ds[0] >= min_frames_per_file for ds in raw_shapes]
    if not all(has_min_frames):
        print(f'The following files will be omitted because they have <{min_frames_per_file} frames:')
        for fp in itertools.compress(filepaths, ~has_min_frames):
            print(f'\t{fp}')
        print()

        filepaths = list(itertools.compress(filepaths, has_min_frames))
        raw_shapes = raw_shapes[has_min_frames,:]

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

class SessionDataset(Dataset):
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
        sequence_key: RNG key
        sequence_length: Sequence length of resulting chunks
        sequence_step_size: Step size between frames in resulting chunks
        min_frames_per_file (int): Minimum number of frames that a file needs to
            have in order to be included. Default: -1, no minimum.
        name: Name of dataset
    """

    filepath: Path
    name: str
    raw_shape: Shape
    sequence_shape: Shape
    total_samples: int
    sequence_slices: List[Slice]

    def __init__(self,
                 filepath: Pathlike,
                 sequence_key: PRNGKey,
                 sequence_length: int=72_000,
                 sequence_step_size: int=1,
                 name: Optional[str]=None,
                 ):

        self.filepath = Path(filepath)
        self.name = self.filepath.parts[-1][:-3] if name is None else name

        with h5py.File(filepath, 'r') as f:
            self.raw_shape = f['stage/block0_values'].shape

        # Get sequence slice indices that define the chunks of this dataset
        # Each file starts at a random index, so that we are less biased in
        # sampling start vs. end of files
        self.sequence_slices = chunk_into_sequences(
            sequence_key, self.raw_shape[0], sequence_length, sequence_step_size)

        self.total_samples = len(self.sequence_slices) * sequence_length

        self.sequence_shape = (sequence_length, *self.raw_shape[1:])

    def __len__(self):
        """Length of dataset is the number of sequence chunks it contains."""
        return len(self.sequence_slices)

    def __getitem__(self, index: int) -> Array:
        """Return specified sequence.

        Returns
            data: shape (sequence_length, data_dim)
        """

        slc = self.sequence_slices[index]
        with h5py.File(self.filepath, 'r') as f:
            data = jnp.asarray(f['stage/block0_values'][slc])

        return data.squeeze()

class MultiessionDataset(ConcatDataset):
    """Multisession dataset as a concatenation of multiple single session datasets.

    Assumes all datasets are created with the same sequence shape

    Args:
        datasets: Iterable of SessionDatasets
    """

    datasets: List[Dataset]
    cumulative_sizes: List[int]
    filepaths: List[Path]
    sequence_shape: Shape
    total_samples: int

    def __init__(self, datasets: Iterable[SessionDataset]):
        super(MultiessionDataset, self).__init__(datasets)

        self.filepaths = [ds.filepath for ds in self.datasets]
        self.total_samples = sum([ds.total_samples for ds in self.datasets])
        self.sequence_shape = self.datasets[0].sequence_shape
    
    @classmethod
    def init_from_paths(cls,
                        filepaths: Sequence[Pathlike],
                        sequence_key: PRNGKey,
                        sequence_length: int=72_000,
                        sequence_step_size: int=1,
                        ) -> 'MultisessionDataset':
        """Create a concatenated dataset of multiple sessions."""

        datasets = [SessionDataset(fp, k, sequence_length, sequence_step_size)
                    for fp, k in zip(filepaths, jr.split(sequence_key, len(filepaths)))]

        return cls(datasets)