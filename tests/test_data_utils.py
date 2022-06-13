"""Test Dataset and Dataloader classes"""

import jax.config
jax.config.update('jax_platform_name', 'cpu')

from absl.testing import absltest
import chex

from kf.data_utils import FishPCDataset

import jax.numpy as np

import os
from os import listdir
from os.path import join, isfile

DATADIR = os.environ['DATADIR']
fish_name = 'fish0_137'

fish_dir = join(DATADIR, fish_name)
filenames = sorted([
    f for f in listdir(fish_dir) if isfile(join(fish_dir, f))
])

class TestPCDataset(chex.TestCase):
    def testDataset(self):
        ds = FishPCDataset(fish_name, DATADIR, min_num_frames=0)

        # Length of num_frames list should equal "length" of dataset
        self.assertTrue(len(ds.num_frames)==len(ds))

        # Number of frames
        days = [2, 29, 73, 113, 179]
        true_num_frames = [1726618, 1725858, 1726768, 1724919, 1726394]
        self.assertTrue(np.all([ds.num_frames[d] for d in days] == true_num_frames))
        
        # Check values for a specific day of recording
        arr42 = ds[42]
        self.assertTrue(ds.num_frames[42]==1726798)
        self.assertTrue(np.all(arr42.shape==(1726798, 15)))

        frames = [557, 1399, 2297, 3271, 4243, 5297, 6317, 7433]
        
        # Retrieved from via df.iloc[frames, :15]
        true_arr = np.array([
            [-1.966646,  0.345493,  0.043362,  0.431899, -0.817103, -0.935952, -0.424814,  0.376971,  0.634887, -0.090397,  0.760033,  0.071908, -0.118746, -0.168039,  0.066190],
            [ 2.615208,  1.025665, -0.066453, -0.477657, -0.482437,  0.861030,  1.129166, -0.180552, -0.757272,  0.098482, -1.435469, -0.230594, -0.243873,  0.693551, -0.407637],
            [-0.902804,  0.334857,  1.022243, -0.166225, -0.576343,  0.244597,  0.479642, -0.327299,  0.698095,  0.089056,  0.248242,  0.241325,  0.407678,  0.138657, -0.235453],
            [14.743798,  3.703990, -2.458721, -1.065586, -0.646491,  0.393797, -1.819442,  3.062200, -0.920465, -0.228458, -1.192837,  3.415640,  0.836353,  4.467017,  0.391304],
            [12.646988,  2.134266, -2.825542, -0.213738, -1.144571, -0.359945,  0.925129,  0.076371, -1.498962,  0.063391, -0.078405,  3.043216, -0.279251,  1.371338, -0.786616],
            [10.047414, -0.259415,  3.082530,  0.382753, -0.576971,  1.417283,  1.518086, -0.198673, -0.973273, -1.250911, -0.740673, -2.170504, -1.460372, -0.503871,  0.235554],
            [ 1.267209,  0.818983,  0.893022, -0.545455,  0.316748,  0.966011, -1.973926, -0.653991,  2.230900,  0.614860, -0.245441,  1.368389,  0.087961,  1.234590, -0.467187],
            [11.345995,  2.287398, -2.228170, -0.294546, -0.225051,  0.383855, -1.956223,  3.434738, -1.336736,  1.779052, -0.474929,  4.321470,  1.996296,  4.421712, -0.477084],
        ])

        self.assertTrue(np.allclose(arr42[frames,:], true_arr, atol=1e-6))

    def testDataSubsetIndex(self):
        """Specify index range of files to use."""

        i_range = (10, 15)
        fnames_ref = ['p3_fish0_137_20210126.h5',
                      'p3_fish0_137_20210127.h5', 
                      'p3_fish0_137_20210128.h5', 
                      'p3_fish0_137_20210129.h5', 
                      'p3_fish0_137_20210130.h5',]

        ds = FishPCDataset(fish_name, DATADIR,
                           data_subset=i_range, min_num_frames=0)
        self.assertTrue(len(ds) == len(fnames_ref),
                        f'Expected {len(fnames_ref)}, got {len(ds)}')
        self.assertTrue(ds.filenames == fnames_ref,
                        'Files and/or file ordering in dataset do not exactly match expected values. ' \
                        + f'Got {ds.filenames}')

    def testDataSubsetFilename(self):
        """Specify filenames in non-sequential order and some that do not exist."""
        
        fnames_input = ['p3_fish0_137_20210413.h5', 
                        'p3_fish0_137_20210311.h5',     # Does not exist
                        'p3_fish0_137_20210205.h5', 
                        'p3_fish0_137_20210811.h5',
                        'p3_fish0_137_20210129.h5',
                        'p3_fish0_137_20210606.h5',]
        
        fnames_ref = ['p3_fish0_137_20210129.h5',
                      'p3_fish0_137_20210205.h5', 
                      'p3_fish0_137_20210413.h5', 
                      'p3_fish0_137_20210606.h5', 
                      'p3_fish0_137_20210811.h5']

        ds = FishPCDataset(fish_name, DATADIR,
                           data_subset=fnames_input, min_num_frames=0)
        self.assertTrue(len(ds) == len(fnames_ref),
                        f'Expected {len(fnames_ref)}, got {len(ds)}')
        self.assertTrue(ds.filenames == fnames_ref,
                        'Files and/or file ordering in dataset do not exactly match expected values. ' \
                        + f'Got {ds.filenames}')

if __name__ == '__main__':
    absltest.main()