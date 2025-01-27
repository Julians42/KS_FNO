from jax_models import *
from KS_solver import KS_step
import numpy as np 
import torch


n_samples = 1200
n_points = 512
l = 2 * np.pi * 32  # domain length
dt = .25  # time step size
modes = 20
T = 2000  # total simulation time
data = np.zeros((n_samples, T, n_points))
width = 512

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

# initialize KS
KS = KuramotoSivashinsky(dt=dt, s=width, l=l, M=modes)

for i in range(n_samples):
    # run KS for n-1 steps (accounting for the initial condition)
    data[i] = KS_step(KS, T-1, f_rand(.1)(np.linspace(0, l, width)))

# save data
fpath = f"/scratch/julian/neuralop/pygen_{n_samples}_{n_points}_{T}_{modes}_{l:.2f}.pth"
torch.save(torch.tensor(data), fpath)
