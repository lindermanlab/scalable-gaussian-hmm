# Time EM on real data

import os

from jax import vmap, pmap, lax
import jax.numpy as np
import jax.random as jr
from functools import partial, reduce

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step as e_step,
                          collective_m_step as m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)

from kf.data_utils import FishPCDataset, FishPCDataloader


DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']

