# Script to evaluate the model based on predefined metrics
# To be run after training the model
# Julian Schmitt 

import sys
import torch
import numpy as np 
import matplotlib.pyplot as plt
import yaml

import neuralop

import evaluation_metrics as em

config = yaml.safe_load(open("ks_config.yaml")) # select the config file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model 
model = torch.load(config["default"]["save_model_path"]).to(device)
# model = torch.load("/scratch/julian/neuralop/ks_models/pygen_1200_512_2000_20_201.06.pth").to(device)

# load the data 
data = torch.load(config["default"]["data"]["folder"] + config["default"]["data"]["file"]).to(torch.float32).permute(2, 1, 0)

# plot a sample prediction and compare to the true data
sample = data[:, :, -2]
FNO_pred = sample.clone().to(device)

# start prediction after spin-up time of 500
for i in range(500, config["default"]["data"]["architecture"]["T"] - 1):
    FNO_pred[:, i+1] = model(FNO_pred[:, i].unsqueeze(0).unsqueeze(0))

# plot the prediction
fig, ax = plt.subplots(1, 2, figsize=(9, 5))

T = config["default"]["data"]["architecture"]["T"]
dt = config["default"]["data"]["architecture"]["dt"]
n_points = config["default"]["data"]["architecture"]["n_points"]
l = config["default"]["data"]["architecture"]["l"]

t_vals = np.linspace(0, T*dt, T)
x_vals = np.linspace(0, l, n_points)

ax[0].pcolormesh(t_vals, 
                 x_vals, 
                 sample, 
                 cmap = "inferno", 
                 vmin = -3, 
                 vmax = 3)
ax[0].set_title("True")
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("space")
ax[1].pcolormesh(t_vals, 
                 x_vals,
                 FNO_pred.to("cpu").detach().numpy(), 
                 vmin = -3, 
                 vmax = 3, 
                 cmap = "inferno")
ax[1].set_title(f"FNO (starting at t=500 * {dt})")
ax[1].set_xlabel("time (s)")
print(torch.mean(sample))
print(torch.mean(FNO_pred))

# save figure to figures folder 
plt.savefig("FNO_pred2.png")
