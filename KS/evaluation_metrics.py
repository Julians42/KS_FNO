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
    umean = np.mean(u, axis=0) 
    return np.mean((u-umean)**2, axis=1)

# compute 1D spectrum 
def spectrum1(data_array, l=22):
    u = np.fft.fft2(data_array)
    k_max = data_array.shape[1] // 2

    spectrum = np.sqrt((np.abs(u)**2).mean(axis=0))[0:k_max + 1]
    freqs = np.fft.fftfreq(u.shape[1]+1, d=l/u.shape[1])[0:u.shape[1]//2+1]
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


def plot_tke_spectra(data_result, nn_result, threshold = 3.2):

    x_vals = np.linspace(0, threshold, 500)

    # compute TKE for both distributions
    tke_data = TKE(data_result)
    tke_nn = TKE(nn_result)

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

def plot_freq_spectrum(data_array, nn_array, l=22):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_yscale('log')

    operator_spectra, freqs = spectrum1(data_array, l)
    nn_spectra, freqs = spectrum1(nn_array, l)

    ax.plot(freqs * l, operator_spectra, label='True', color='k')
    ax.plot(freqs * l, nn_spectra, label='NN', color='r', linestyle = '--')
    ax.legend()
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Energy')
    ax.set_title('1D Spectrum')
    return fig, np.max(np.abs(operator_spectra - nn_spectra)), freqs[np.argmax(np.abs(operator_spectra - nn_spectra))]

def plot_autocorr(data_result, nn_result):
    truth_autocorr_list = np.zeros(data_result.shape)
    nn_autocorr_list = np.zeros(nn_result.shape)
    for index in range(data_result.shape[1]):
        truth_autocorr_list[:, index] = np.correlate(data_result[:,index], data_result[:,index], mode='full')[len(data_result[:,index])-1:]
        nn_autocorr_list[:, index] = np.correlate(nn_result[:,index], nn_result[:,index], mode='full')[len(nn_result[:,index])-1:]
    # return truth_autocorr_list, nn_autocorr_list
    # plot means  
    fig = plt.figure(figsize=(8,8))
    plt.plot(np.mean(truth_autocorr_list, axis=1), label='True', color='k')
    plt.plot(np.mean(nn_autocorr_list, axis=1), label='NN', color='r', linestyle = '--')

    plt.legend()
    plt.ylabel("Autocorrelation")
    plt.xlabel("Lag");

    # compute MSE and add to title 
    mse = np.mean((np.mean(truth_autocorr_list, axis=1) - np.mean(nn_autocorr_list, axis=1))**2)
    plt.title(f"Autocorrelation Comparison with MSE: {mse:.2g}");
    return fig, mse


# plot comparison of actual trajectory
def plot_trajectory_comparison(data_result, nn_result):
    t = np.arange(0, KS.dt*(num_steps+1-0.1), KS.dt)

    vmax = np.max(data_result)
    vmin = np.min(data_result)

    fig, ax = plt.subplots(2,1, figsize=(12,8))

    cm = ax[0].pcolormesh(t, x_plot, data_result[:-1].T, cmap='magma', vmax=vmax, vmin=vmin)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('x')
    ax[0].set_title('Target')
    plt.colorbar(cm, ax=ax[0])

    ax[1].pcolormesh(t, x_plot, nn_result[:-1].T, cmap='magma', vmax=vmax, vmin=vmin)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('x')
    ax[1].set_title('neural network')
    plt.colorbar(cm, ax=ax[1])

    plt.subplots_adjust(hspace=0.4)


def evaluate_metrics(true_data, nn_data, save_to_pdf=False, l = 22, tke_plot_max_threshold = 3.2):
    fig1, ks_density = plot_histogram_density(true_data, nn_data);
    fig2, ks_tke = plot_tke_spectra(true_data, nn_data, tke_plot_max_threshold);
    fig3, max_error, freq_max_error = plot_freq_spectrum(true_data, nn_data, l);
    fig4, mse = plot_autocorr(true_data, nn_data)

    # print metrics of interest
    print(f"Histogram Kolmogorov-Smirnov Error: {ks_density.statistic:.2g}, p-value: {ks_density.pvalue:.2g}")
    print(f"TKE Kolmogorov-Smirnov Error: {ks_tke.statistic:.2g}, p-value: {ks_tke.pvalue:.2g}")
    print(f"Max Error in Spectrum: {max_error:.2g} at frequency {freq_max_error:.2g}")
    print(f"Mean Squared Error in Autocorrelation: {mse:.2g}")

    if  save_to_pdf != False:
        with PdfPages(f'{save_to_pdf}.pdf') as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
    return fig1, fig2, fig3, fig4, ks_density, ks_tke
