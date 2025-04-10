# SimPEG functionality
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg.utils import plot_1d_layer_model, download, mkvc
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

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
#%matplotlib inline
mpl.rcParams.update({"font.size": 12})

import time
import contextlib
import warnings


#Import helper functions
import sys
import data_functions, inversion_functions

# Optional: create a no-op output stream to silence function printouts
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

warnings.filterwarnings("ignore")

# Create function of inversion:
def halfspace_inversion(line_station, freq_avoid = [8200., 40000., 135000.], uncertainty_floor = 5.0e0, relative_error = 0.05, layer_thick_halfspace = [1000], n_layers_halfspace = 1, alpha_s = 1e-5, alpha_x = 1):

    ## Calculate the height 

    #Calculate the height (terrain clearance)
    line_station.loc['height_tx'] = line_station['gpsz_tx'] - line_station['dtm']

    ### Survey ### No
    survey = inversion_functions.survey_object(line_station, freq_avoid)

    ### Data ###
    data_object = inversion_functions.data_object(line_station, relative_error = relative_error, noise_floor = uncertainty_floor, survey = survey, freq_avoid = freq_avoid)

    ### Mesh and Mapping ###    Si
    regularization_mesh_hs, log_conductivity_halfspace_map, simulation_hsp_L2 = inversion_functions.mapping_forward_objects(layer_thicknesses = layer_thick_halfspace, n_layers = n_layers_halfspace, survey = survey)

    ### Starting Models ###
    
    # Starting model is log-conductivity values (S/m)
    starting_conductivity_model_hsp = np.log(1e-3 * np.ones(n_layers_halfspace))
    
    # Reference model, same as starting 
    reference_conductivity_model_hsp = starting_conductivity_model_hsp.copy()

    ### Data Misfit ###
    dmis_hsp_L2 = data_misfit.L2DataMisfit(simulation=simulation_hsp_L2, data=data_object)

    ### Regularization ### No
    reg_L2 = inversion_functions.regularization_object(regularization_mesh_hs, reference_conductivity_model = reference_conductivity_model_hsp , alpha_s = alpha_s, alpha_x = alpha_x)

    ### Optimization ### Si
    opt_L2 = optimization.InexactGaussNewton(maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3)

    ### Inversion Parameters ###
    inv_L2 = inversion_functions.inversion_setup(dmis_hsp_L2, reg_L2, opt_L2)

    ### Run inversion ###
    recovered_halfspace_model_L2 = inv_L2.run(starting_conductivity_model_hsp)

    ### Get recovered model ###
    ## Get the recovered halfspace resistivity from model estimated
    conductivities_hsp = log_conductivity_halfspace_map * recovered_halfspace_model_L2
    resistivities_hsp = 1 / conductivities_hsp

    return conductivities_hsp

#### Reading Data ####

# Get the current working directory (where the notebook is running)
current_dir = os.getcwd()

# Move one level up to the parent directory
parent_dir = os.path.dirname(current_dir)

# Construct the full path to the CSV file in the parent directory
csv_path = os.path.join(parent_dir, "block53_fdem_inv.csv")

fdem_data = pd.read_csv(csv_path) ## Set to the position of the csv_files list where the database you want is
pd.set_option('display.max_columns', len(fdem_data.columns))
fdem_data.head()

# Define like working with:
line_no = 'L530100'
line = fdem_data.loc[fdem_data['Line'] == line_no].copy()

# Calculate the tx_rx coordinates:
line = data_functions.calculate_tx_rx_coordinates(line)

#### Loop through soundings ####

# Make loop to run for every single fid in data

# Filter out NaN fid values
line = line[line['fid'].notna()]

# Get valid fid values
line_stations = line.fid.values
n_stations = len(line_stations)
n_layers = 1

# Preallocate output array
results = np.zeros((len(line_stations), n_layers + 1))  # +1 for fid

# Start global timer
start_time = time.time()

# Loop through each fid
for i, fid_n in enumerate(line_stations):

    # Get the row for this station
    line_station = line.loc[line.fid == fid_n].iloc[0]

    # Calculate terrain clearance (height of tx above surface)
    height_tx = line_station['gpsz_tx'] - line_station['dtm']

    # Print progress every 100 stations
    if (i + 1) % 10 == 0 or (i + 1) == len(line_stations):
        elapsed = time.time() - start_time
        print(f"Processed {i + 1} of {len(line_stations)} stations "
              f"(elapsed time: {elapsed:.2f}s)")

    ### Run the inversion ### (without printing stuff)
    with contextlib.redirect_stdout(DummyFile()):
        conductivities_hsp = halfspace_inversion(line_station)
    
    ### Save fid and conductivities ###
    results[i, 0] = fid_n
    results[i, 1:] = conductivities_hsp

### Save datafile ###

# Create column names
column_names = ['fid'] + [f'cond_{i+1}' for i in range(n_layers)]

# Convert to DataFrame
df = pd.DataFrame(results, columns=column_names)

# Save to CSV
csv_filename = f"outputs/{line_no}_halfspace_conductivities.csv"
df.to_csv(csv_filename, index=False)
print(f"\n CSV saved to {csv_filename}")