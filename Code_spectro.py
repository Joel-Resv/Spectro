
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

def plot_spectrometer_data(file_path_gin, file_path_vide, label_gin, label_vide):
    # Load data using NumPy
    data_gin = np.loadtxt(file_path_gin)
    data_vide = np.loadtxt(file_path_vide)

    # Extract y and x points from the loaded data for gin and vide
    y_points_gin = data_gin[0, :]
    x_points_gin = data_gin[1, :]

    y_points_vide = data_vide[0, :]
    x_points_vide = data_vide[1, :]

    # Normalize y points by dividing by the max value for gin and vide
    y_points_normalized_gin = y_points_gin / np.max(y_points_gin)
    y_points_normalized_vide = y_points_vide / np.max(y_points_vide)

    # Find peaks in the spectrum of vide
    peaks_vide, _ = find_peaks(y_points_normalized_vide, height=0.1)

    # Subtract the spectrum of vide from the spectrum of gin to remove noise
    y_points_cleaned = y_points_gin - y_points_vide
    
    y_points_cleaned_normalised = y_points_cleaned/np.max(y_points_cleaned)

    # Plot the cleaned data with smaller marker size and add a label
    plt.plot(x_points_gin, y_points_cleaned_normalised, label=label_gin + ' - ' + label_vide, marker='', linestyle='-', markersize=3)

    # Plot the baseline as a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', label='Baseline')

# Provide the paths to your .txt files for gin and vide
file_path_gin = 'spectre_gin'
file_path_vide = 'spectre_vide'

# Call the function to plot the cleaned data
plot_spectrometer_data(file_path_gin, file_path_vide, 'Spectre Gin', 'Spectre Vide')

# Customize the plot
plt.title('Spectre de Raman: Gin sans bruit')
plt.xlabel(r'cm$^{-1}$')
plt.ylabel('Intensité normalisée')
plt.legend()
plt.show()

from scipy.optimize import minimize_scalar

#def als_baseline(y, lam, p, niter=10):
#    L = len(y)
#    D = np.diff(np.eye(L), 2)
#    w = np.ones(L)
#    for i in range(niter):
#        W = np.diag(w)
#        Z = np.diff(np.eye(L), 2)
#        Z = Z[:, 1:-1]  # Adjusting the size of Z to match the size of y
#        Z = Z.T @ Z
#        C = np.linalg.cholesky(Z + lam * W)
#        ZS = np.linalg.solve(C, Z.T)
#        ZSZ = ZS @ ZS.T
#        y_hat = ZS @ np.linalg.solve(C, y)
#        w = p * (y - y_hat) ** 2
#        w = 1 / (1 + np.exp(-(w - np.percentile(w, 25)) / (np.percentile(w, 75) - np.percentile(w, 25))))
#    return y_hat

#def plot_spectrometer_data(file_path_gin, file_path_vide, label_gin, label_vide):
    # Load data using NumPy
#    data_gin = np.loadtxt(file_path_gin)
#    data_vide = np.loadtxt(file_path_vide)#

    # Extract y and x points from the loaded data for gin and vide
#    y_points_gin = data_gin[0, :]
#    x_points_gin = data_gin[1, :]

#    y_points_vide = data_vide[0, :]
#    x_points_vide = data_vide[1, :]

    # Normalize y points by dividing by the max value for gin and vide
#    y_points_normalized_gin = y_points_gin / np.max(y_points_gin)
#    y_points_normalized_vide = y_points_vide / np.max(y_points_vide)

    # Subtract the spectrum of vide from the spectrum of gin to remove noise
#    y_points_cleaned = y_points_gin - y_points_vide
#    y_points_cleaned_normalised = y_points_cleaned / np.max(y_points_cleaned)

    # Perform baseline correction using the asymmetric least squares algorithm
#    lam = 1e6
#    p = 0.1
#    baseline = als_baseline(y_points_cleaned_normalised, lam, p)

    # Subtract the baseline from the cleaned spectrum
#    y_points_baseline_subtracted = y_points_cleaned_normalised - baseline

    # Plot the cleaned data with smaller marker size and add a label
#    plt.plot(x_points_gin, y_points_baseline_subtracted, label=label_gin + ' - ' + label_vide, marker='', linestyle='-', markersize=3)

    # Plot the baseline as a horizontal line at y=0
#    plt.axhline(y=0, color='gray', linestyle='--', label='Baseline')

# Provide the paths to your .txt files for gin and vide
#file_path_gin = 'spectre_99%'
#file_path_vide = 'spectre_vide'

# Call the function to plot the cleaned data
#plot_spectrometer_data(file_path_gin, file_path_vide, 'Spectre Gin', 'Spectre Vide')

# Customize the plot
#plt.title("Spectre de Raman de l'éthanol 99% avec baseline")
#plt.xlabel(r'cm$^{-1}$')
#plt.ylabel('Intensité normalisée')
#plt.legend()
#plt.show()
