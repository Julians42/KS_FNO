import sys
import torch
import numpy as np 
import matplotlib.pyplot as plt
import yaml
import neuralop
import evaluation_metrics2 as em
from neuralop.models.fno import FNO  # Ensure FNO class is imported
import os


def coarsen_data(data, factor):
    return data[:,:, ::factor]

torch.serialization.add_safe_globals([FNO])  # Explicitly allow FNO

path = f'/central/groups/esm/reusebi/MLStability/KS_FNO'
config = yaml.safe_load(open(f"{path}/KS/ryan_config.yaml")) # select the config file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model_names = os.listdir(f"{path}/KS/models")
for model_name in model_names:
    print(model_name)
    modelpath = f"{path}/KS/models/{model_name}"
    coarsef = int(modelpath.split('coarsen')[1].split('_')[0])
    tpoints = int(modelpath.split('tpoints')[1].split('.')[0])

    savepath = f'{path}/KS/pdfs/{model_name.replace(".pth", "")}'
    if os.path.exists(f"{savepath}.pdf"):
        print(savepath, ' exists! Moving on...')
        continue
    # load the model 
    model = torch.load(modelpath, map_location=device, weights_only=False).to(device)
    # model = torch.load("/scratch/julian/neuralop/ks_models/pygen_1200_512_2000_20_201.06.pth").to(device)

    # load the data 
    data = torch.load(config["default"]["data"]["folder"] + config["default"]["data"]["file"]).to(torch.float32)
    # plot a sample prediction and compare to the true data
    n_test = 16
    sample = data[-n_test:, :, :]
    sample = coarsen_data(sample, coarsef)
    FNO_pred = sample.clone()
    del(data)

    # start prediction after spin-up time of 500
    with torch.no_grad():
        for i in range(500, config["default"]["data"]["architecture"]["T"] - 501):
            x = torch.stack([FNO_pred[:,i+k+1-tpoints,:] for k in range(tpoints)], axis=1).to(device)
            out = model(x).squeeze().detach().cpu()
            print(out.max())
            FNO_pred[:,i+1, :] = out

    dt = 0.25
    l = config['default']['data']['architecture']['l']
    dx = l/(512/coarsef + 1)

    FNO_pred = FNO_pred[:,501:,:].cpu().detach().numpy()
    sample = sample[:,501:,:].cpu().detach().numpy()

    em.evaluate_metrics(sample, FNO_pred, dt, dx, save_to_pdf=savepath)
    del(model)
    del(sample)
    del(FNO_pred)
    print(f"saved pdf to {savepath}!")


