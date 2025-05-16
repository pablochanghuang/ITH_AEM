# discretize functionality
from discretize import TensorMesh

# Common Python functionality
import os
import numpy as np
import pandas as pd
from scipy.constants import mu_0
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import ipywidgets
import dill
from simpeg.utils import plot_1d_layer_model, download, mkvc

mpl.rcParams.update({"font.size": 12})


### Plotting functions ###
def plot_inv_results(regularization_mesh, conductivities, dpred_l2, dobs=None, frequencies = np.r_[135, 40, 8.2, 1.8, 0.4]*1e3):


    # Create one figure with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left subplot: 1D Layer Model ---
    ax1 = axes[0]
    plot_1d_layer_model(
        regularization_mesh.h[0],
        conductivities,
        ax=ax1,
        show_layers=False,
        color="b"
    )
    ax1.set_title("Recovered Model")
    ax1.legend(["Model"])

    # --- Right subplot: Predicted Data ---
    ax2 = axes[1]
    # Uncomment and add observed data if needed
    if dobs is not None:
        ax2.loglog(frequencies, np.abs(dobs[0::2]), "k-o", label="Observed (real)")
        ax2.loglog(frequencies, np.abs(dobs[1::2]), "k:o", label="Observed (imag)")
    ax2.loglog(frequencies, np.abs(dpred_l2[0::2]), "b-o", label="Model (real)")
    ax2.loglog(frequencies, np.abs(dpred_l2[1::2]), "b:o", label="Model (imag)")

    ax2.set_xlabel("Frequencies (Hz)")
    ax2.set_ylabel("|Hs/Hp| (ppm)")
    ax2.set_title("Predicted Data (Model)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
    
#Calculate cummulative distances
def calculate_cumulative_distance(df, x_col='x_tx', y_col='y_tx'):
    """
    Calculate the cumulative distance from the first point in a profile.

    Parameters:
    df (pd.DataFrame): The dataframe containing the X and Y coordinates.
    x_col (str): The name of the column containing the X coordinates. Default is 'X'.
    y_col (str): The name of the column containing the Y coordinates. Default is 'Y'.

    Returns:
    pd.DataFrame: The input dataframe with an additional column 'dist' containing the cumulative distances.
    """
    # Extract the X and Y coordinates
    x = df[x_col].values
    y = df[y_col].values
    
    # Calculate the differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate the Euclidean distance between consecutive points
    distances = np.sqrt(dx**2 + dy**2)
    
    # Prepend a 0 to the distances array to represent the distance from the first point to itself
    distances = np.insert(distances, 0, 0)
    
    # Calculate the cumulative distance
    cumulative_distances = np.cumsum(distances)
    
    # Add the cumulative distances to the dataframe
    df['dist'] = cumulative_distances
    
    return df

# Plot tikhonov curves, based on Lindsey Heagy EOSC556 2024 code:
def plot_tikhonov_curve(beta_values, phi_d, phi_m, phid_star=None, highlight_iter=None):
    """
    Plot Tikhonov trade-off curves showing the relationship between
    regularization strength (beta), data misfit (phi_d), and model norm (phi_m).
    
    Parameters:
        beta_values (array-like): List or array of beta values.
        phi_d (array-like): List or array of data misfit values.
        phi_m (array-like): List or array of model norm values.
        phid_star (float, optional): Target misfit line to show on the plots.
        highlight_iter (int, optional): Iteration index to highlight on all plots.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot phi_d vs beta
    ax[0].loglog(beta_values, phi_d, marker='o')
    ax[0].set_xlabel(r'$\beta$')
    ax[0].set_ylabel(r'Data Misfit $\phi_d$')
    ax[0].invert_xaxis()
    if phid_star:
        ax[0].axhline(phid_star, linestyle='--', color='k', label=r'Target Data Misfit $\phi_d^*$')
        ax[0].legend()

    # Plot phi_m vs beta
    ax[1].loglog(beta_values, phi_m, marker='o')
    ax[1].set_xlabel(r'$\beta$')
    ax[1].set_ylabel(r'Data Misfit $\phi_m$')
    ax[1].invert_xaxis()

    # Plot phi_d vs phi_m (Tikhonov curve)
    ax[2].loglog(phi_m, phi_d, marker='o')
    ax[2].set_xlabel(r'Model Norm $\phi_m$')
    ax[2].set_ylabel(r'Data Misfit $\phi_d$')
    if phid_star:
        ax[2].axhline(phid_star, linestyle='--', color='k', label=r'Target Data Misfit $\phi_d^*$')
        ax[2].legend()

    # Highlight a specific iteration if given
    if highlight_iter is not None:
        ax[0].plot(beta_values[highlight_iter], phi_d[highlight_iter], 'ro')
        ax[1].plot(beta_values[highlight_iter], phi_m[highlight_iter], 'ro')
        ax[2].plot(phi_m[highlight_iter], phi_d[highlight_iter], 'ro')

    fig.suptitle("Tikhonov Curves", fontsize=14)
    plt.tight_layout()
    return fig, ax


#Calculate the coordinates of transmitter and receiver
def calculate_tx_rx_coordinates(line, plot_histogram=False):
    """
    Calculates the transmitter (Tx) and receiver (Rx) coordinates for various frequencies
    based on azimuth values from the input dataframe.
    
    Parameters:
    - line (pd.DataFrame): Input dataframe containing x_tx and y_tx columns.
    - plot_histogram (bool): Whether to plot a histogram of azimuth values. Default is False.
    
    Returns:
    - pd.DataFrame: Updated dataframe with calculated Tx and Rx coordinates.
    """
    theta = []  # To store the co-azimuth in radians
    Az_rad = []  # To store the azimuth values in radians

    # Dictionary with Tx-Rx coil separation
    d = {
        'cp140k': 7.95,
        'cp40k': 7.93,
        'cp8200': 7.95,
        'cx3300': 9.06,
        'cp1800': 7.94,
        'cp400': 7.93
    }

    # Initialize arrays for Tx and Rx coordinates
    num_points = line.x_tx.values.shape[0]
    coords = {f"{key}_{suffix}": np.zeros(num_points) for key in d for suffix in ["tx_x", "tx_y", "rx_x", "rx_y"]}

    # Calculate coordinates based on azimuth
    for i in range(num_points - 1):
        deltay = line.y_tx.values[i + 1] - line.y_tx.values[i]
        deltax = line.x_tx.values[i + 1] - line.x_tx.values[i]
        # angle = np.arctan2(deltay, deltax)
        angle = np.arctan(deltay / deltax)
        theta.append(angle)
        # print(f'angle = {angle}')
        Az_rad.append(angle if angle >= 0 else 2 * np.pi + angle)

        for key, dist in d.items():
            dx = (dist / 2) * np.sin(angle)
            dy = (dist / 2) * np.cos(angle)

            coords[f"{key}_tx_x"][i] = line.x_tx.values[i] + dx
            coords[f"{key}_tx_y"][i] = line.y_tx.values[i] + dy
            coords[f"{key}_rx_x"][i] = line.x_tx.values[i] - dx
            coords[f"{key}_rx_y"][i] = line.y_tx.values[i] - dy

    Az_rad = np.array(Az_rad)
    Az_deg = Az_rad * 180 / np.pi

    if plot_histogram:
        plt.hist(Az_deg, bins=100)
        plt.title("Azimuth Distribution (Degrees)")
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Frequency")
        plt.show()

        print(f'Mean azimuth: {np.mean(Az_deg):.2f} degrees')
        print(f'Std azimuth: {np.std(Az_deg):.2f} degrees')
        print('-----------------------------------\n')

    # Assign calculated Tx-Rx coordinates to the dataframe
    for key, values in coords.items():
        line[key] = values

    return line