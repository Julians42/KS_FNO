import sys
import argparse

import numpy as np
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
from neuralop.models import FNO2d

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

# import custom data loader from tensor.py
from tensor import TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}!")

# get the config file to be run from the command line
# parser = argparse.ArgumentParser()
# parser.add_argument("--config_file", type=str, required=True, help="Path to the config YAML file")
# args = parser.parse_args()

# config_file = args.config_file
config_file = "/home/jschmitt/KS_FNO/FNO2D/architecture/config.yml"

print(f"Using config file: {config_file}")


pipe = ConfigPipeline(
    [
        YamlConfig(
            config_file, config_name = "default", config_folder = "."
        ),
        ArgparseConfig(infer_types = True, config_name = None, config_file = None),
        YamlConfig(config_folder = ".")
    ]
)

config = pipe.read_conf()


# Load in data - currently messy format - would be good to separate into different file or 
# reproduce our data generation process in python, but leaving as this for now ;)  


# f = h5py.File(config.data.folder + "KS.mat")
# data = torch.tensor(f['u'][:], dtype = torch.float32)
data = torch.load(config.data.folder + config.data.file).to(torch.float32).permute(2, 1, 0)

def coarsen_data(data, factor):
    return data[::factor, :, :]

data = coarsen_data(data, config.data.coarsen_factor)

####
width, time, samples = data.shape

n_history = config.fno2d.in_channels
n_train = config.data.n_train
n_test = config.data.n_test 
data_start = config.data.data_start

T = time - (n_history - 1)

X_samp = []
y_samp = []
for sample in range(samples):
    for window_start in range(data_start, 2000 - n_history, 10): # start, stop, step (step could make sense to be n_history)
        window_end = window_start + n_history
        if (window_end+1 >= 2000) or (len(X_samp) >= (n_train + n_test)):
            # Exit if we have enough samples or if the window exceeds the data length
            break
        X_samp.append(data[:, window_start:window_end, sample])
        y_samp.append(data[:, window_end + 1, sample])


# Stack along a new axis for time history
Xtrain = torch.stack(X_samp[:n_train], dim=0) 
ytrain = torch.stack(y_samp[:n_train], dim=0)
Xtest = torch.stack(X_samp[n_train:], dim=0)
ytest = torch.stack(y_samp[n_train:], dim=0)

# expand dims to be (num_training_samples, 1, width, n_history)
Xtrain = Xtrain.unsqueeze(1)  # (n_train, 1, width, n_history)
ytrain = ytrain.unsqueeze(1)  # (n_train, 1, width)
Xtest = Xtest.unsqueeze(1)  # (n_test, 1, width, n_history)
ytest = ytest.unsqueeze(1)  # (n_test, 1, width)

# shp_old_Xtrain = data[:, (data_start - 1):1999, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)
# shp_ytrain = data[:, data_start:2000, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)


#####
# Rearrange to (samples, timesteps, width)
# Xtrain = Xtrain.permute(3, 2, 1, 0)  # (n_train, T, 3, 512)
# Xtrain = Xtrain.reshape(-1, n_history, width)  # (n_train*(T), 3, 512)

# Xtrain = data[:, (data_start - 1):1999, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)
# ytrain = data[:, data_start:2000, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)

# n_test = config.data.n_tests[0] # could implement for different resolutions
# Xtest = data[:, (data_start - 1):1999, (1200 - n_test):].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)
# ytest = data[:, data_start:2000, (1200 - n_test):].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)

# data loaders
# Define training dataset
train_dataset = TensorDataset(
    x=Xtrain,
    y=ytrain,
    transform_x=None,
    transform_y=None
)

# Define testing dataset
test_dataset = TensorDataset(
    x=Xtest,
    y=ytest,
    transform_x=None,
    transform_y=None
)

# Create DataLoaders
batch_size = config.data.batch_size
train_loader_ks = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader_ks = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)
# singular test loader for now
test_loaders = {256: test_loader_ks}


# Creating l2 and h10 loss functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )

# Load model from configuration
operator = get_model(config)

optimizer = AdamW(operator.parameters(), 
                  lr=config.opt.learning_rate, 
                  weight_decay=config.opt.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.opt.step_size,
    gamma=config.opt.gamma
)

print("Model has {} parameters".format(count_model_params(operator)))

# train the model 
trainer = Trainer(
    model = operator, 
    n_epochs = config.opt.n_epochs,
    device = device,

    verbose = True
)

trainer.train(
    train_loader = train_loader_ks,
    test_loaders = test_loaders,
    optimizer = optimizer,
    scheduler = scheduler,
    regularizer = False,
    save_every = 5,
    save_dir = "./checkpoints",
)
torch.save(operator, config.save_model_path)
print("Training complete!")
