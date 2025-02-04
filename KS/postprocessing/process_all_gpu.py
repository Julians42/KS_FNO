# post processing 


import neuralop
import numpy as np
import matplotlib.pyplot as plt


import sys
import torch

import yaml
import gc
import imp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import evaluation_metrics as em

for i in [2,4,8,16,32,64]:
    model = torch.load(f"/central/groups/esm/jschmitt/neuralop/ks_models/res_{i}.pth", weights_only=False) # , map_location=torch.device('cpu')
    config = yaml.safe_load(open(f"/home/jschmitt/KS_FNO/KS/resolution_configs/resolution_{i}.yaml"))

    if i ==1:
        # for some reason this model is better
        model = torch.load(f"/central/groups/esm/jschmitt/neuralop/ks_models/pygen_full_res.pth", weights_only=False) # , map_location=torch.device('cpu')

    data = torch.load("/central/groups/esm/jschmitt/neuralop/pygen_200_512_2000_20_201.06.pth").to(device).to(torch.float32).permute(2, 1, 0)
    data = data[:, :, 0:2] # only use first 2 observations for debugging, used for plotting comparison

    data_coarse = data[::i, :, :] # coarsen data for prediction
    print("Data shape ", data_coarse.shape)


    # make predictions
    predictions = torch.zeros(data_coarse.shape)

    with torch.no_grad():
        for i in range(data_coarse.shape[2]):
            print("Element ", i)
            pred_i = data_coarse[:, :, i].clone().to(device)
            
            for j in range(500, 1999):
                pred_i[:, j+1] = model(pred_i[:, j].unsqueeze(0).unsqueeze(0)).detach()

            predictions[:, :, i] = pred_i.cpu()#.unsqueeze(2)

    em.evaluate_metrics(data.detach().cpu().numpy(), predictions.detach().cpu().numpy(), config, f"res_{str(i)}")

    print("Done with ", i)