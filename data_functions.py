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

mpl.rcParams.update({"font.size": 12})


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