import torch
import numpy as np
import matplotlib.pyplot as plt
# import h5py
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import yaml
import evaluation_metrics2 as em
import generate_ks_data as gkd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_file = "../architecture/config.yml" # "/home/jschmitt/KS_FNO/FNO2D/architecture/config.yml"

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

model_path = config.save_model_root + f"fno_coarsen{config.data.coarsen_factor}_tpoints{config.t_points}.pth"
model = torch.load(model_path)
model.eval()


# get data
# X, y = gkd.load_data(config, rand = True, n_rand_samples = 20)
data = torch.tensor(gkd.get_samples(config, 20))
# coarsen data
def coarsen_data(data, factor):
    return data[:, :, ::factor]

data = coarsen_data(data, config.data.coarsen_factor)

model = torch.load(model_path)
model.eval().cuda()

###################  Single  ################################
# Prepare
# sample = data[0, :, :].permute(1, 0).float().to(device)
# sample_preds = sample.clone()

# # Forward pass
# with torch.no_grad():
#     for i in range(500, config.data.architecture.T):
#         X_data = sample_preds[:, (i - config.t_points):i]
#         sample_preds[:, i] = model(X_data.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[:, -1]

# # plot sample_preds and save

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# ax[0].pcolormesh(sample.cpu().numpy())
# ax[0].set_title("True")
# ax[1].pcolormesh(sample_preds.detach().cpu().numpy())
# ax[1].set_title("FNO (starting at 500)")

# plt.savefig("ks_fno.png")


# em.evaluate_metrics(sample.cpu(),
#                     sample_preds.cpu(),
#                     config.data.architecture.dt,
#                     config.data.architecture.l / config.data.architecture.width,
#                     save_to_pdf=False)


#####################  Multiple  ##############################

# repeat but for the entire dataset
sample = data.permute(0, 2, 1).unsqueeze(1).float().to(device)
sample_preds = sample.clone()

# Forward pass
with torch.no_grad():
    for i in range(500, config.data.architecture.T):
        X_data = sample_preds[:, :, :, (i - config.t_points):i]
        out = model(X_data)
        # print(out.shape)
        # print(out[:, :, :, -1].shape)
        # print(sample_preds[:, :, :, i].shape)
        sample_preds[:, :, :, i] = out[:, :, :, -1]


# plot sample_preds and save

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].pcolormesh(sample[0, 0, :, :].cpu().numpy())
ax[0].set_title("True")
ax[1].pcolormesh(sample_preds[0, 0, :, :].detach().cpu().numpy())
ax[1].set_title("FNO (starting at 500)")

plt.savefig("ks_fno.png")


# out = model(sample_preds[:, 500:510].unsqueeze(0).unsqueeze(0))

# Evaluate metrics
# em.evaluate_metrics(sample, sample_preds, save_to_pdf=True)
import importlib
importlib.reload(em)
em.evaluate_metrics(sample.squeeze(1).permute(1, 2, 0).cpu(),
                    sample_preds.squeeze(1).permute(1, 2, 0).cpu(),
                    config,
                    save_to_pdf=f"fno_coarsen{config.data.coarsen_factor}_tpoints{config.t_points}")
