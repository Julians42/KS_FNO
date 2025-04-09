# Ryan/Julian ML Models Evaluation Metrics
# Provide true_data and nn_data from Kuramoto Sivashinsky model
# Run as follows: 
# evaluate_metrics(true_data, nn_data, save_to_pdf=True)
# returns plots and metrics for comparison

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde, ks_2samp
from matplotlib.backends.backend_pdf import PdfPages

######### Helper Functions #########
# run KS test for two samples
def ks_test(true_data, nn_data):
    """Computes the Kolmogorov-Smirnov test statistic between two datasets"""
    ks_stat = ks_2samp(true_data, nn_data)
    return ks_stat

# compute TKE
def TKE(u):
    """Computes the total kinetic energy of the"""
    umean = np.mean(u, axis=(1,2)).reshape(-1,1,1) 
    return ((u-umean)**2).flatten()

# compute 1D spectrum 
# ensemble should be axis 0
def spectrum1(data_array, dx, ax):
    u = np.fft.fft(data_array, axis=ax)
    print(u.shape)
    k_max = data_array.shape[2] // 2

    if ax == 1:
        axmean = (0,2)
    if ax == 2:
        axmean = (0,1)
    spectrum = (np.abs(u)**2).mean(axis=axmean)[0:k_max + 1]
    freqs = np.arange(0,len(spectrum))
    return spectrum, freqs

######### Plot Functions #########
# Density Histogram
def plot_histogram_density(data_result, nn_result):
    data_values = data_result.flatten()
    nn_values = nn_result.flatten()

    x_vals = np.linspace(-3.2, 3.2, 500)

    # compute KDE for both distributions
    kde_data = gaussian_kde(data_values)
    kde_nn = gaussian_kde(nn_values)

    # evaluate the KDEs on the x values
    kde_data_values = kde_data(x_vals)
    kde_nn_values = kde_nn(x_vals)

    fig = plt.figure(figsize=(7,6))
    plt.plot(x_vals, kde_data_values, label='True', color='k')
    plt.plot(x_vals, kde_nn_values, label='NN', color='r', linestyle = '--')

    plt.ylabel("Density")
    plt.xlabel("Value")
    plt.legend()

    # compute KS statistic to add to title
    ks_stat = ks_test(data_values, nn_values)
    plt.title(f"Empirical Density Comparison with KS: {ks_stat.statistic:.2g}, p-value: {ks_stat.pvalue:.2g}")
    return fig, ks_stat


def plot_tke_density(data_result, nn_result, threshold = 3.2):

    x_vals = np.linspace(0, threshold, 500)

    # compute TKE for both distributions
    tke_data = TKE(data_result)
    tke_nn = TKE(nn_result)

    print(tke_data.shape)

    # compute KDE for both distributions
    kde_data = gaussian_kde(tke_data)
    kde_nn = gaussian_kde(tke_nn)

    # evaluate the KDEs on the x values
    kde_data_values = kde_data(x_vals)
    kde_nn_values = kde_nn(x_vals)

    fig = plt.figure(figsize=(7,6))
    plt.plot(x_vals, kde_data_values, label='True', color='k')
    plt.plot(x_vals, kde_nn_values, label='NN', color='r', linestyle = '--')

    plt.ylabel("Density")
    plt.xlabel("Energy")
    plt.legend()

    # compute KS statistic to add to title
    ks_stat = ks_test(tke_data, tke_nn)
    plt.title(f"TKE Density Comparison with KS: {ks_stat.statistic:.2g}, p-value: {ks_stat.pvalue:.2g}")
    return fig, ks_stat

def plot_freq_spectrum(data_array, nn_array, dx):
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_yscale('log')

    operator_spectra, freqs = spectrum1(data_array, dx, 2)
    nn_spectra, freqs = spectrum1(nn_array, dx, 2)

    ax.plot(freqs, operator_spectra, label='True', color='k')
    ax.plot(freqs, nn_spectra, label='NN', color='r', linestyle = '--')
    ax.legend()
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Energy')
    ax.set_title('1D Spectrum Space')
    return fig, np.max(np.abs(operator_spectra - nn_spectra)), freqs[np.argmax(np.abs(operator_spectra - nn_spectra))]

def plot_freq_spectrum_time(data_array, nn_array, dt):
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_yscale('log')

    operator_spectra, freqs = spectrum1(data_array, dt, 1)
    nn_spectra, freqs = spectrum1(nn_array, dt, 1)

    ax.plot(freqs * 2 * np.pi, operator_spectra, label='True', color='k')
    ax.plot(freqs * 2 * np.pi, nn_spectra, label='NN', color='r', linestyle = '--')
    ax.legend()
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Energy')
    ax.set_title('1D Spectrum Time')
    return fig, np.max(np.abs(operator_spectra - nn_spectra)), freqs[np.argmax(np.abs(operator_spectra - nn_spectra))]

def plot_spatial_autocorr(data_result, nn_result, dx):
    nx = data_result.shape[-1]
    x_values = np.arange(0, nx*dx, dx)
    truth_autocorr_list = np.zeros(data_result.shape)
    nn_autocorr_list = np.zeros(nn_result.shape)
    for i in range(data_result.shape[0]):
        for t in range(data_result.shape[1]):
            truth_autocorr_list[i,t,:] = np.correlate(data_result[i,t,:], data_result[i,t,:], mode='full')[len(data_result[i,t])-1:]
            nn_autocorr_list[i,t,:] = np.correlate(nn_result[i,t,:], nn_result[i,t,:], mode='full')[len(data_result[i,t])-1:]
    # return truth_autocorr_list, nn_autocorr_list
    # plot means  
    fig = plt.figure(figsize=(6,6))
    truth_autocorr = np.mean(truth_autocorr_list, axis=(0,1))
    nn_autocorr = np.mean(nn_autocorr_list, axis=(0,1))
    plt.plot(x_values, truth_autocorr, label='True', color='k')
    plt.plot(x_values, nn_autocorr, label='FNO', color='r', linestyle = '--')

    plt.legend()
    plt.ylabel("Spatial Autocorrelation")
    plt.xlabel("Lag")
    plt.xlim([0,100])

    # compute MSE and add to title 
    mse = np.mean((truth_autocorr -nn_autocorr)**2)
    plt.title(f"Autocorrelation Comparison with MSE = {mse:.2g}");
    return fig, mse


# plot comparison of actual trajectory
def plot_trajectory_comparison(data_result, nn_result, config):
    T = config["default"]["data"]["architecture"]["T"]
    dt = config["default"]["data"]["architecture"]["dt"]
    n_points = config["default"]["data"]["architecture"]["n_points"]
    l = config["default"]["data"]["architecture"]["l"]
    coarsen_factor = config["default"]["data"]["coarsen_factor"]

    t_vals = np.linspace(0, T*dt, T)
    x_vals = np.linspace(0, l, n_points // coarsen_factor)

    fig, ax = plt.subplots(1, 2, figsize=(9, 5))

    ax[0].pcolormesh(t_vals, 
                    x_vals, 
                    data_result, 
                    cmap = "inferno", 
                    vmin = -3, 
                    vmax = 3)
    ax[0].set_title("True")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("space")
    ax[1].pcolormesh(t_vals, 
                    x_vals,
                    nn_result, 
                    vmin = -3, 
                    vmax = 3, 
                    cmap = "inferno")
    ax[1].set_title(f"FNO (starting at t=500 * {dt})")
    ax[1].set_xlabel("time (s)")

    return fig

def plot_temporal_autocorr(data_result, nn_result, dt):
    nt = data_result.shape[1]
    x_values = np.arange(0, nt*dt, dt)
    truth_autocorr_list = np.zeros(data_result.shape)
    nn_autocorr_list = np.zeros(nn_result.shape)
    for i in range(data_result.shape[0]):
        for x in range(data_result.shape[2]):
            truth_autocorr_list[i,:,x] = np.correlate(data_result[i,:,x], data_result[i,:,x], mode='full')[len(data_result[i,:,x])-1:]
            nn_autocorr_list[i,:,x] = np.correlate(nn_result[i,:,x], nn_result[i,:,x], mode='full')[len(data_result[i,:,x])-1:]
    # return truth_autocorr_list, nn_autocorr_list
    # plot means  
    fig = plt.figure(figsize=(6,6))
    truth_autocorr = np.mean(truth_autocorr_list, axis=(0,2))
    nn_autocorr = np.mean(nn_autocorr_list, axis=(0,2))
    plt.plot(x_values, truth_autocorr, label='True', color='k')
    plt.plot(x_values, nn_autocorr, label='FNO', color='r', linestyle = '--')

    plt.legend()
    plt.ylabel("Spatial Autocorrelation")
    plt.xlabel("Lag")
    plt.xlim([0,100])

    # compute MSE and add to title 
    mse = np.mean((truth_autocorr -nn_autocorr)**2)
    plt.title(f"Autocorrelation Comparison with MSE = {mse:.2g}");
    return fig, mse



def evaluate_metrics(true_data, nn_data, dt, dx, save_to_pdf=False, tke_plot_max_threshold = 3.2):


    # fig0 = plot_trajectory_comparison(true_data, nn_data, config)
    fig1, ks_density = plot_histogram_density(true_data, nn_data)
    fig2, ks_tke = plot_tke_density(true_data, nn_data, tke_plot_max_threshold)
    fig3, max_error, freq_max_error = plot_freq_spectrum(true_data, nn_data, dx)
    fig4, max_error2, freq_max_error2 = plot_freq_spectrum_time(true_data, nn_data, dt)
    fig5, mse = plot_spatial_autocorr(true_data, nn_data, dx)
    fig6, mse_temporal = plot_temporal_autocorr(true_data, nn_data, dt)

    # print metrics of interest
    print(f"Histogram Kolmogorov-Smirnov Error: {ks_density.statistic:.2g}, p-value: {ks_density.pvalue:.2g}")
    print(f"TKE Kolmogorov-Smirnov Error: {ks_tke.statistic:.2g}, p-value: {ks_tke.pvalue:.2g}")
    print(f"Max Error in Spectrum: {max_error:.2g} at frequency {freq_max_error:.2g}")
    print(f"Mean Squared Error in Autocorrelation: {mse:.2g}")

    if  save_to_pdf != False:
        with PdfPages(f'{save_to_pdf}.pdf') as pdf:
            # pdf.savefig(fig0)
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
            pdf.savefig(fig5)
            pdf.savefig(fig6)
    return fig1, fig2, fig3, fig4, fig5, ks_density, ks_tke
