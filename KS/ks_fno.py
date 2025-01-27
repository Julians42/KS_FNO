#%%
import sys
import scipy
import torch
import h5py
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, WeightedSumLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.utils import get_wandb_api_key, count_model_params

from torch.utils.data import DataLoader

#%%


#%%
data_fpath = "/scratch/julian/neuralop/KS.mat"
ks_data = scipy.io.loadmat(data_fpath)
f = h5py.File(data_fpath)
# load as torch object - permute to be in (samples, width, length) format
data = torch.tensor(f['u'][:], dtype = torch.float32)

# define a data loader object
Xtrain = data[:, 99:1999, :960]
ytrain = data[:, 100:2000, :960]

Xtest = data[:, 99:1999, 960:]
ytest = data[:, 99:1999, 960:]


#%%

