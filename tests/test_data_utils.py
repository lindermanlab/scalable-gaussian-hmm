"""Test Dataset and Dataloader classes

To run all tests:
    python test_data_utils.py
To run single class:
    python test_data_utils.py TestClassName
To run single test:
    python test_data_utils.py TestClassName.testMethodName
"""

from absl.testing import absltest

import os
import chex
import jax.numpy as np
import jax.random as jr

from kf.data_utils import (FishPCDataset, FishPCDataloader,
                           arg_uniform_split)
                           

DATADIR = os.environ['DATADIR']
fish_name = 'fish0_137'

class TestPCDataset(chex.TestCase):
    def testBasic(self):
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

    def testSubsetByIndex(self):
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

    def testSubsetByFilename(self):
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

    def testTrainTestSplit(self,):
        """Split dataset into train and test subsets, deterministic."""
        
        # First 5 files and 6th file, respectively
        train_ref = ['p3_fish0_137_20210116.h5',
                     'p3_fish0_137_20210117.h5',
                     'p3_fish0_137_20210118.h5',
                     'p3_fish0_137_20210119.h5',
                     'p3_fish0_137_20210120.h5',]
        test_ref  = ['p3_fish0_137_20210121.h5']

        ds = FishPCDataset(fish_name, DATADIR, data_subset='all', min_num_frames=0)
        # --------------------
        # Specify via integers
        num_train = 5
        num_test = 1
        train_ds, test_ds = ds.train_test_split(num_train=num_train, num_test=num_test)

        self.assertTrue(len(train_ds) == num_train)
        self.assertTrue(len(test_ds) == num_test)

        self.assertTrue(train_ds.filenames == train_ref)
        self.assertTrue(test_ds.filenames == test_ref)

        # -----------------------------------
        # Specify via mix of frac and integer. Value of frac_train specified 
        # such that int(207 * 0.025) ==5, so we can reuse above references
        frac_train = 0.025 
        train_ds, test_ds = ds.train_test_split(frac_train=frac_train, num_test=num_test)
        self.assertTrue(train_ds.filenames == train_ref)
        self.assertTrue(test_ds.filenames == test_ref)

    def testTrainTestSplitRandom(self,):
        """Split dataset into train and test subsets, randomized."""

        seed = jr.PRNGKey(519)

        # First 5 files and 6th file, respectively
        # In this test, make sure that datasets are NOT these files
        train_ref = ['p3_fish0_137_20210116.h5',
                     'p3_fish0_137_20210117.h5',
                     'p3_fish0_137_20210118.h5',
                     'p3_fish0_137_20210119.h5',
                     'p3_fish0_137_20210120.h5',]
        test_ref  = ['p3_fish0_137_20210121.h5']

        ds = FishPCDataset(fish_name, DATADIR, data_subset='all', min_num_frames=0)
        # --------------------
        # Specify via integers
        num_train = 5
        num_test = 1
        train_ds, test_ds = \
            ds.train_test_split(num_train=num_train, num_test=num_test, seed=seed)

        self.assertEqual(len(train_ds), num_train)
        self.assertEqual(len(test_ds), num_test)

        self.assertNotEqual(train_ds.filenames, train_ref)
        self.assertNotEqual(test_ds.filenames, test_ref)

        # -----------------------------------
        # Specify via mix of frac and integer. Value of frac_train specified 
        # such that int(207 * 0.025) ==5, so we can reuse above references
        frac_train = 0.025 
        train_ds, test_ds = ds.train_test_split(
            frac_train=frac_train, num_test=num_test, seed=seed)
        self.assertNotEqual(train_ds.filenames, train_ref)
        self.assertNotEqual(test_ds.filenames, test_ref)

class TestPCDataloader(chex.TestCase):
    def testShapeFullDays(self,):
        """Test length and batch shapes when loading full days of data."""

        ds = FishPCDataset(fish_name, DATADIR, data_subset='all', min_num_frames=1700000)

        num_frames_per_batch = int(np.min(ds.num_frames))
        dl = FishPCDataloader(ds, batch_size=1, num_frames_per_batch=num_frames_per_batch).__iter__()
        
        self.assertEqual(len(dl), len(ds))
        self.assertEqual(dl.batch_shape, (1, num_frames_per_batch, ds.dim),
                         f'Expected batch shape {(1, num_frames_per_batch, ds.dim)}, received {dl.batch_shape}.')

    def testShapeSplitDays(self,):
        """Test length and batch shapes when loading days of data in batches."""

        ds = FishPCDataset(fish_name, DATADIR, data_subset=(0,4), min_num_frames=1700000)

        num_frames_per_batch = 400000 # Therefore, expect 4 batches from a day of data
        dl = FishPCDataloader(ds, batch_size=-1, num_frames_per_batch=num_frames_per_batch)

        self.assertEqual(len(dl), 1,
                         f'Loading all data at once. Expected Dataloader length 1, received {len(dl)}')
        self.assertEqual(dl.batch_shape, (len(ds)*4, num_frames_per_batch, ds.dim),
                         f'Expecting batch shape {(len(ds)*4, num_frames_per_batch, ds.dim)}, received {dl.batch_shape}')

    def testBatchBasic(self,):
        """Test Dataloader loading correct batches of data."""
        # Load first 3 files, modify Dataset to think files only have ~100 frames
        ds = FishPCDataset(fish_name, DATADIR, data_subset=(0,3), min_num_frames=0)
        ds._num_frames = np.array([98, 123, 100])
        
        # So, we have set sizes of sum([4,6,5]) = 15
        # Which results in 3 batches out of this iteration, where
        num_frames_per_batch = 20
        batch_size = 5
        batch_shape = (batch_size, num_frames_per_batch, ds.dim)

        dl = FishPCDataloader(ds,
                                batch_size=batch_size,
                                num_frames_per_batch=num_frames_per_batch)
        self.assertEqual(len(dl), 3)

        dl = dl.__iter__()
        ref = np.empty(batch_shape)

        # batch 0 should consist of ds[0][0:80] + ds[1][0:20]
        b = dl.__next__()
        ref = np.concatenate([ds[0][0:80], ds[1][0:20]]).reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 1 should consist of ds[1][20:120]
        b = dl.__next__()
        ref = ds[1][20:120].reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 2 should consist of ds[2][0:100]
        b = dl.__next__()
        ref = ds[2][0:100].reshape(*batch_shape)
        
        self.assertTrue(np.all(b==ref))
    
    def testBatchShuffle(self,):
        """Test Dataloader loading correct batches of data when shuffled."""
        # Load first 3 files, modify Dataset to think files only have ~100 frames
        ds = FishPCDataset(fish_name, DATADIR, data_subset=(0,4), min_num_frames=0)
        ds._num_frames = np.array([98, 123, 100, 131])
        
        # So, we have set sizes of sum([4,6,5,6]) = 21
        # Which results in 4 batches out of this iteration, where
        num_frames_per_batch = 20
        batch_size = 5
        batch_shape = (batch_size, num_frames_per_batch, ds.dim)

        dl = FishPCDataloader(ds,
                                batch_size=batch_size,
                                num_frames_per_batch=num_frames_per_batch,
                                shuffle=True,
                                seed=jr.PRNGKey(34002))
        self.assertEqual(len(dl), 4)

        # ---------------------------------------------------
        # First iter: datafiles will be permuted into [1,2,3,0]
        dl = dl.__iter__()

        ref = np.empty(batch_shape)
        # batch 0 should consist of ds[1][0:100]
        b0 = dl.__next__()
        ref = ds[1][0:100].reshape(*batch_shape)
        self.assertTrue(np.all(b0==ref))

        # batch 1 should consist of ds[1][100:120] + ds[2][0:80]
        b = dl.__next__()
        ref = np.concatenate([ds[1][100:120], ds[2][0:80]]).reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 2 should consist of ds[2][80:100] + ds[3][0:80]
        b = dl.__next__()
        ref = np.concatenate([ds[2][80:100], ds[3][0:80]]).reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 3 should consist of ds[3][80:120] + ds[0][0:60]
        b = dl.__next__()
        ref = np.concatenate([ds[3][80:120], ds[0][0:60]]).reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # ---------------------------------------------------
        # Second iter: datafiles will be permuted into [1,0,2,3]
        dl = dl.__iter__()
        self.assertEqual(dl._shuffle_count, 2)

        bb0 = dl.__next__()
        self.assertTrue(np.all(b0==bb0))

        # batch 1 should consist of ds[1][100:120] + ds[0][0:80]
        b = dl.__next__()
        ref = np.concatenate([ds[1][100:120], ds[0][0:80]]).reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 2 should consist of ds[2][0:100]
        b = dl.__next__()
        ref = ds[2][0:100].reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

        # batch 3 should consist of ds[3][0:100]
        b = dl.__next__()
        ref = ds[3][0:100].reshape(*batch_shape)
        self.assertTrue(np.all(b==ref))

    # TODO
    # def testSpeckle(self,):
    #     """Randomly select frames within a day of data."""
    #     # Create a dataset....  
    #     num_frames_per_day = 10
    #     batch_size = 5

    #     dl = FishPCDataloader(ds,
    #                           batch_size=batch_size,
    #                           num_frames_per_day=num_frames_per_day).__iter__()
    #     dl_speckled = FishPCDataloader(ds,
    #                                    batch_size=batch_size,
    #                                    num_frames_per_day=num_frames_per_day,
    #                                    speckle=True, seed=self.seed).__iter__()

    #     data = next(dl)    
    #     data_speckled = next(dl_speckled)
    #     self.assertFalse(np.allclose(data, data_speckled))
        
class TestMiscFns(chex.TestCase):
    def testUniformSplit(self,):
        """Use elements from multiple sets."""
        # Original sets = [0,1,2,3], [4], [5,6,7], [8,9], [10,11,12,13]
        original_sets = np.split(np.arange(14), (4,5,8,10))
        batch_size = 3

        # Get args to split, then split original set
        original_set_sizes = list(map(len, original_sets))
        num_batches = sum(original_set_sizes) // batch_size
        args = arg_uniform_split(batch_size, original_set_sizes)
        uniform_sets = np.empty((num_batches, batch_size))
        for i_batch, b_args in enumerate(args):
            uniform_sets = \
                uniform_sets.at[i_batch] \
                            .set(np.concatenate([original_sets[i_set][s_set[0]:s_set[1]] \
                                                 for (i_set, s_set) in b_args]))

        expected_sets = np.arange(12).reshape(4, 3)
        self.assertTrue(np.all(uniform_sets==expected_sets))
    
    def testUniformSplit2(self,):
        """Middle sets have clean finishes."""
        # Original sets: [0,1,2,3], [4,5,6,7,8,9], ...
        #                [10,11,12,13,14], [15,16,17,18,19,20,21]
        original_sets = np.split(np.arange(22), (4,10,15))
        batch_size = 5

        # Get args to split, then split original set
        original_set_sizes = list(map(len, original_sets))
        num_batches = sum(original_set_sizes) // batch_size
        args = arg_uniform_split(batch_size, original_set_sizes)
        uniform_sets = np.empty((num_batches, batch_size))
        for i_batch, b_args in enumerate(args):
            uniform_sets = \
                uniform_sets.at[i_batch]\
                            .set(np.concatenate([original_sets[i_set][s_set[0]:s_set[1]] \
                                                 for (i_set, s_set) in b_args]))

        expected_sets = np.arange(20).reshape(4,5)

        self.assertTrue(np.all(uniform_sets==expected_sets))

    
if __name__ == '__main__':
    absltest.main()