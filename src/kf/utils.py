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
    'kmeans_initialization',
    'CheckpointDataclass',
]

def kmeans_initialization(seed, num_states, dataloader, step_size=1200,
                          emission_covs_scale=1.):
    """Initalize a GaussianHMM using k-means algorithm.
    
    Args:
        seed (jr.PRNGKey):
        num_states (int): Number of clusters to fit
        dataloader (torch.utils.data.Dataloader):
        step_size (int): Number of frames between selected frames of a sequence,
            a larger value results in greater subsampling. Choose large enough
            that we get meaninful data reduction but not so small that kmeans
            fit takes too long. Default: 1200, corresponding to 1 fr/min @ 20 Hz
            (assuming that dataloader sequences are NOT subsampled)
        emission_covs_scale (float or None): Scale of emission covariances
            initialized to block identity matrices. If None, bootstrap emission
            covariances from kmeans labels.
    """
    
    seed_kmeans, seed_initial, seed_transition = jr.split(seed, 3)

    # Get single batch from dataloader and pre-allocate array
    _batch = next(iter(dataloader))
    batch_size, seq_length, dim = _batch.shape
    subsampled = onp.empty((len(dataloader), batch_size, seq_length//step_size, dim))

    # Get data from dataloader, and reshape to a 2-d array
    for i, batch_emissions in enumerate(dataloader):
        subsampled[i] = batch_emissions[...,::step_size,:]
    subsampled = subsampled.reshape(-1, dim)

    # Print out some stats
    train_emissions = len(dataloader) * batch_size * seq_length
    print(f'Fitting k-means with {len(subsampled)}/{train_emissions} frames, ' + \
          f'{len(subsampled)/train_emissions*100:.2f}% of training data...' + \
          f'Subsampled at {step_size / 60 / 20:.2f} frames / min.')

    # Set emission means and covariances based on fitted k-means clusters
    kmeans = KMeans(num_states, random_state=int(seed_kmeans[-1])).fit(subsampled)
    emission_means = jnp.asarray(kmeans.cluster_centers_)

    if emission_covs_scale is None:
        labels = kmeans.labels_
        emission_covs = onp.stack([
            jnp.cov(subsampled[labels==state], rowvar=False) for state in range(num_states)
        ])
    else: 
        emission_covs = jnp.tile(jnp.eye(dim) * emission_covs_scale, (num_states, 1, 1))

    # Randomly set initial state and state transition probabilities
    initial_probs = jr.dirichlet(seed_initial, jnp.ones(num_states))
    transition_matrix = jr.dirichlet(seed_transition, jnp.ones(num_states), (num_states,))

    return GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

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
            hmm = GaussianHMM(**{k: jnp.asarray(f[k]) for k in hmm_keys})
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
            out = (None, -1, None)
        
        return (*out, last_ckp_path) if return_path else out