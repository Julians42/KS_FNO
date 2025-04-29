# Generate KS data for the FNO2D model using a config file to specify the architecture and parameters.
import sys
sys.path.append("../evaluation")
from jax_models import *
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

import numpy as np
import torch
import yaml

def KS_step(KS, num_steps, y0, all_steps=True):
    y_full = np.zeros((num_steps+1, len(y0)))
    y_full[0] = y0
    for i in range(num_steps):
        y_full[i+1] = KS.step(y_full[i])

    if all_steps:
        return y_full
    else:
        return y_full[-1]


def f_rand(L):
    """Initial Condition"""
    nwaves = 5
    a = np.arange(nwaves)+1
    b = np.random.rand(nwaves)
    d = np.random.rand(nwaves)

    xi = np.linspace(0,L,1000)
    fx = xi*0
    for i in range(nwaves):
        fx = fx + b[i]*np.sin(2*a[i]*np.pi*(xi-d[i])/L)

    mina = np.min(fx)
    maxa = np.max(fx)


    def f(x):
        fx = x*0
        for i in range(nwaves):
            fx = fx + b[i]*np.sin(2*a[i]*np.pi*(x-d[i])/L)

        fx = (fx - mina) / (maxa - mina)

        # compute the mean evaluated at 1000 points
        mean_fx = np.trapz(fx, np.linspace(0, L, len(fx))) / L

        # center around 0 for zero average energy
        fx = fx - mean_fx
        return fx

    return f

def coarsen_data(data, factor):
    return data[::factor, :, :]

def get_samples(config, n_samples):
    # unpack the necessary parameters from the config
    n_points = config.data.architecture.n_points
    l = config.data.architecture.l
    dt = config.data.architecture.dt
    T = config.data.architecture.T
    dt = config.data.architecture.dt
    width = config.data.architecture.width
    modes = config.data.architecture.modes

    # initialize KS
    KS = KuramotoSivashinsky(dt=dt, s=width, l=l, M=modes)
    data = np.zeros((n_samples, T, n_points))
    for i in range(n_samples):
        # run KS for n-1 steps (accounting for the initial condition)
        data[i] = KS_step(KS, T-1, f_rand(.1)(np.linspace(0, l, width)))

    return data


def load_data(config, rand = True, n_rand_samples = 10):
    """Load the data or generate random samples with rand = True with n_rand_samples"""
    data = None
    if not rand:
        data = torch.load(config.data.folder + config.data.file).to(torch.float32).permute(2, 1, 0)
        data = coarsen_data(data, config.data.coarsen_factor)
    if rand:
        # generate new random data for testing
        data = get_samples(config, n_rand_samples)
    # ensure data is a torch tensor
    data = torch.tensor(data, dtype=torch.float32)


    ####
    width, time, samples = data.shape

    n_history = config.history_length
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
            y_samp.append(data[:, (window_start+1):(window_end+1), sample])

    if not rand:
        # Stack along a new axis for time history
        Xtrain = torch.stack(X_samp[:n_train], dim=0)
        ytrain = torch.stack(y_samp[:n_train], dim=0)
        Xtest = torch.stack(X_samp[n_train:], dim=0)
        ytest = torch.stack(y_samp[n_train:], dim=0)

        # expand dims to be (num_training_samples, 1, width, n_history)
        Xtrain = Xtrain.unsqueeze(1)
        ytrain = ytrain.unsqueeze(1)
        Xtest = Xtest.unsqueeze(1)
        ytest = ytest.unsqueeze(1)

        return Xtrain, ytrain, Xtest, ytest
    else:
        X = torch.stack(X_samp, dim=0)
        y = torch.stack(y_samp, dim=0)
        # expand dims to be (num_training_samples, 1, width, n_history)
        X = X.unsqueeze(1)
        y = y.unsqueeze(1)
        return X, y



# test functions for a given config
if __name__ == "__main__":
    # get the config file to be run from the command line

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

    # generate data
    n_samples = 200
    data = get_samples(config, n_samples)
    print(f"Generated {n_samples} samples of data with shape {data.shape}")


    # test the load_data function with rand = True to generate new data and rand = False to load training data

    X, y = load_data(config, rand = True, n_rand_samples = 10)
    print(f"Loaded {X.shape[0]} samples of data with shape {X.shape}")
    Xtrain, ytrain, Xtest, ytest = load_data(config, rand = False)
    print(f"Loaded {Xtrain.shape[0]} training samples with shape {Xtrain.shape}")
