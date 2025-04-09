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
from neuralop.models import FNO

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

# import custom data loader from tensor.py
from tensor import TensorDataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}!")

# config_file = args.config_file
config_file = "/central/groups/esm/reusebi/MLStability/KS_FNO/KS/ryan_config.yaml"

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
arch = config['arch']
coarsef = config['data']['coarsen_factor']
# config[arch]['n_modes'] = [round(512/coarsef)]
config[arch]['train_resolution'] = [round(512/coarsef)]
config[arch]['test_resolutions'] = [round(512/coarsef)]

n_modes = config[arch]['n_modes']
epochs = config['opt']['n_epochs']
hidden_channels = config[arch]['hidden_channels']
nlayers = config[arch]['n_layers']


# Load in data - currently messy format - would be good to separate into different file or 
# reproduce our data generation process in python, but leaving as this for now ;)  

data = torch.load(config.data.folder + config.data.file).to(torch.float32)
def coarsen_data(data, factor):
    return data[:, :, ::factor]

data = coarsen_data(data, config.data.coarsen_factor)

n_train = config.data.n_train
data_start = config.data.data_start
Xtrain = data[:n_train, (data_start - 1):1999, :].reshape(-1,data.shape[-1]).unsqueeze(1)
ytrain = data[:n_train, data_start:2000, :].reshape(-1,data.shape[-1]).unsqueeze(1)

n_test = config.data.n_tests[0] # could implement for different resolutions
Xtest = data[(1200 - n_test):, (data_start - 1):1999, :].reshape(-1,data.shape[-1]).unsqueeze(1)
ytest = data[(1200 - n_test):, data_start:2000, :].reshape(-1,data.shape[-1]).unsqueeze(1)


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
test_loaders = {512/coarsef: test_loader_ks}


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

save_model_path = f'/central/groups/esm/reusebi/MLStability/KS_FNO/KS/models/fno_coarsen{coarsef}_nmodes{n_modes}_epochs{epochs}_hchannels{hidden_channels}_nlayers{nlayers}.pth'
torch.save(operator, save_model_path)
print("Training complete!")
