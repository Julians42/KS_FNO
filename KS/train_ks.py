import sys
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


# Load in model configuration
config_name = "default"

pipe = ConfigPipeline(
    [
        YamlConfig(
            "ks_config.yaml", config_name = "default", config_folder = "."
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

n_train = config.data.n_train
data_start = config.data.data_start
Xtrain = data[:, (data_start - 1):1999, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)
ytrain = data[:, data_start:2000, :n_train].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)

n_test = config.data.n_tests[0] # could implement for different resolutions
Xtest = data[:, (data_start - 1):1999, (1200 - n_test):].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)
ytest = data[:, data_start:2000, (1200 - n_test):].flatten(1, -1).unsqueeze(1).permute(2, 1, 0)

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
test_loaders = {512: test_loader_ks}


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
