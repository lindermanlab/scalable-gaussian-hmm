# Helper classes for managing killifish data

import os
import glob
from dataclasses import dataclass

import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from sklearn.cluster import KMeans

from ssm_jax.hmm.models import GaussianHMM

__all__ = [
    'CheckpointDataclass',
]

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
        emission_dim = hmm.emission_means.value.shape[-1]
        onp.savez_compressed(
            ckp_path,
            initial_probabilities = hmm.initial_probs.value,
            transition_matrix = hmm.transition_matrix.value,
            emission_means = hmm.emission_means.value,
            emission_covariance_matrices = hmm.emission_covariance_matrices.value,
            emission_prior_mean = hmm._emission_prior_mean.value,
            emission_prior_concentration = hmm._emission_prior_conc.value,
            emission_prior_scale = hmm._emission_prior_scale.value,
            emission_prior_extra_df = hmm._emission_prior_df.value - emission_dim,
            epochs_completed = this_epoch,
            **arrs)
        
        return ckp_path
    
    def load(self, path):
        """Load checkpoint from specified path."""

        hmm_keys = ['initial_probabilities',
                    'transition_matrix',
                    'emission_means',
                    'emission_covariance_matrices',
                    'emission_prior_mean',
                    'emission_prior_concentration',
                    'emission_prior_scale',
                    'emission_prior_extra_df',
                    ]
        
        with onp.load(path) as f:
            epochs_completed = f['epochs_completed']
            hmm = GaussianHMM(**{k: jnp.asarray(f[k]) for k in hmm_keys})
            all_train_lps = f['all_train_lps'] if 'all_train_lps' in f else None
            all_test_lps = f['all_test_lps'] if 'all_test_lps' in f else None
            
        return hmm, epochs_completed, all_train_lps, all_test_lps
    
    def load_latest(self):
        """Load latest checkpoint (alphanumerically) in this instance's directory.
        
        Returns:
            hmm (GaussianHMM)
            prev_epoch (int):
            prev_train_lps (array-like or None):
            prev_test_lps (array-like or None):
            ckp_path (str): Path of latest file
        """

        existing_ckps = sorted(self._get_existing_files())
        
        if existing_ckps:
            last_ckp_path = existing_ckps[-1]
            out = self.load(existing_ckps[-1])
        else:   # No checkpoints in this directory
            last_ckp_path = None
            out = (None, -1, None, None)
        
        return (*out, last_ckp_path)