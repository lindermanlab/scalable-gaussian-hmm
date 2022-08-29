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
    'train_and_checkpoint',
]

def kmeans_initialization(seed, num_states, dataset, step_size=1200, seq_length=180, emission_covs_scale=1.):
    """Initalize a GaussianHMM using k-means algorithm.
    
    Args:
        seed (jr.PRNGKey):
        num_states (int): Number of clusters to fit
        dataset (torch.utils.data.Dataset):
        step_size (int): Number of frames between selected frames of a sequence,
            a larger value results in greater subsampling. Choose large enough
            that we get meaninful data reduction but not so small that kmeans
            fit takes too long. Default: 1200, corresponding to 1 fr/min @ 20 Hz.
        seq_length (int): Number of frames per sequence. Choose it to be large
            enough such that data is accessed efficiently, but not so large that
            lots of sequences may be dropped if they are cannot reach minimum
            length before end of file. Default: 180, which corresponds to seqs
            spanning 3 hrs.
        emission_covs_scale (float or None): Scale of emission covariances
            initialized to block identity matrices. If None, bootstrap emission
            covariances from kmeans labels.

    Returns:
        GaussianHMM
    """
    # TODO seed_data not used...
    seed_data, seed_kmeans, seed_initial, seed_transition = jr.split(seed, 4)

    # Subsample data at specific step_size and for specific sequence lenth
    # TODO need to set seq_length*step_size when creating dataset...
    seq_slices = dataset.slice_seq(seq_length, step_size, drop_incomplete_seqs=True)
    emissions = onp.stack([dataset[slc] for slc in seq_slices], axis=0)
    emissions = emissions.reshape(-1, dataset.dim)

    print(f'Fitting k-means with {len(emissions)}/{len(dataset)} frames, ~{len(emissions)/len(dataset)*100:.2f}% of dataset...')
    print(f'subsampled at {step_size / 60 / 20:.2f} frames / min. sequences span {step_size*seq_length/60/60:.1f} hr')

    # Set emission means and covariances based on fitted k-means clusters
    kmeans = KMeans(num_states, random_state=int(seed_kmeans[-1])).fit(emissions)
    emission_means = jnp.asarray(kmeans.cluster_centers_)

    if emission_covs_scale is None:
        labels = kmeans.labels_
        emission_covs = onp.stack([
            jnp.cov(emissions[labels==state], rowvar=False) for state in range(num_states)
        ])
    else: 
        emission_covs = jnp.tile(jnp.eye(dataset.dim) * emission_covs_scale, (num_states, 1, 1))

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