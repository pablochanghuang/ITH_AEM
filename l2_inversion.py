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
def l2_inversion(line_station, conductivities_hsp, freq_avoid = [], uncertainty_floor = 5.0e0, relative_error = 0.05, n_layers = 2, depth_max = 500, depth_min = 1, geometric_factor = 1.2, alpha_s = 1e-5, alpha_x = 1, single_beta = None, beta_ratio = 1e1, inv_prob_return = False):

    ## Calculate the height 

    #Calculate the height (terrain clearance)
    line_station.loc['height_tx'] = line_station['gpsz_tx'] - line_station['dtm']

    ### Survey ### No
    survey = inversion_functions.survey_object(line_station, freq_avoid)

    ### Data ###
    data_object = inversion_functions.data_object(line_station, relative_error = relative_error, noise_floor = uncertainty_floor, survey = survey, freq_avoid = freq_avoid)

    ### Mesh and Mapping ### 
    regularization_mesh_L2, log_conductivity_L2_map, simulation_L2, n_layers = inversion_functions.mapping_forward_objects(n_layers = n_layers, survey = survey, depth_max = depth_max, depth_min = depth_min, geometric_factor = geometric_factor)

    ### Starting Models ###
    
    # Starting model is log-conductivity values (S/m) from halfspace results
    starting_conductivity_model_L2 = np.log(conductivities_hsp * np.ones(n_layers))

    # Reference model is also log-resistivity values (S/m)
    reference_conductivity_model_L2 = starting_conductivity_model_L2.copy()

    ### Data Misfit ###
    dmis_hsp_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=data_object)

    ### Regularization ### No
    reg_L2 = inversion_functions.regularization_object(regularization_mesh_L2, reference_conductivity_model = reference_conductivity_model_L2 , alpha_s = alpha_s, alpha_x = alpha_x)

    ### Optimization ### Si
    opt_L2 = optimization.InexactGaussNewton(
    maxIter=200, maxIterLS=20, maxIterCG=20, tolCG=1e-3
    )

    ### Inversion Parameters ###
    # Combine the inverse problem and the set of directives
    if inv_prob_return:
        inv_L2, inv_prob = inversion_functions.inversion_setup(dmis_hsp_L2, reg_L2, opt_L2, single_beta = single_beta, beta_ratio = beta_ratio, inv_prob_return = inv_prob_return)
    else: 
        inv_L2 = inversion_functions.inversion_setup(dmis_hsp_L2, reg_L2, opt_L2, single_beta = single_beta, beta_ratio = beta_ratio, inv_prob_return = inv_prob_return)

    ### Run inversion ###
    recovered_model_L2 = inv_L2.run(starting_conductivity_model_L2)

    ### Get recovered model ###
    ## Get the recovered halfspace resistivity from model estimated
    conductivities_L2 = log_conductivity_L2_map * recovered_model_L2
    #resistivities_l2 = 1 / conductivities_l2

    if inv_prob_return:
        return regularization_mesh_L2, log_conductivity_L2_map, simulation_L2, inv_L2, inv_prob, recovered_model_L2, conductivities_L2, data_object
    else: 
        return conductivities_L2

def main(line_no = 'L530030', single_beta=5, conductivities_hsp = 0.03422309704591167, save_file = True):
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
    line = fdem_data.loc[fdem_data['Line'] == line_no].copy()

    # Calculate the tx_rx coordinates:
    line = data_functions.calculate_tx_rx_coordinates(line)

    # Calculate the cumulative distance
    line = data_functions.calculate_cumulative_distance(line)

    # Filter out NaN fid values
    line = line[line['fid'].notna()]

    # Calculate mean through fid values: 2 soundings per fid
    numeric_filtered_line = line.select_dtypes(include=["number"])
    line = numeric_filtered_line.groupby("fid", as_index=False).mean()


    #### Set halspace conductivities ####

    # Set Halfspace conductivity:
    resistivities_hsp = 1/conductivities_hsp


    #### Loop through soundings ####

    # Make loop to run for every single fid in data

    # Get valid fid values
    line_stations = line.fid.values
    n_stations = len(line_stations)

    # Preallocate output array
    results = []  # +1 for fid

    # Start global timer
    start_time = time.time()

    # Loop through each fid
    for i, fid_n in enumerate(line_stations):

        # Get the row for this station
        line_station = line.loc[line.fid == fid_n].iloc[0]

        # Calculate terrain clearance (height of tx above surface)
        line_station.loc['height_tx'] = line_station['gpsz_tx'] - line_station['dtm']

        ### Calculate max depth ###
        min_frequency = 400
        skin_depth = 503*np.sqrt(resistivities_hsp/np.min(min_frequency))
        depth_max = 2*skin_depth  # depth to lowest layer

        # Print progress every 100 stations
        if (i + 1) % 10 == 0 or (i + 1) == len(line_stations):
            elapsed = time.time() - start_time
            print(f"Processed {i + 1} of {len(line_stations)} stations "
                f"(elapsed time: {elapsed:.2f}s)")

        ### Run the inversion ### (without printing stuff)
        with contextlib.redirect_stdout(DummyFile()):
            conductivities_l2 = l2_inversion(line_station, conductivities_hsp, depth_max = depth_max, single_beta = single_beta)
        
        ### Save fid and conductivities ###
        #results[i, 0] = fid_n
        #results[i, 1:] = conductivities_l2

        row = [fid_n] + [line_station.dist] + [line_station.dtm] + list(conductivities_l2)  # model can be any-length array
        results.append(row)

    ### Save datafile ###

    # Create column names
    n_layers = max(len(row) - 3 for row in results)
    column_names = ['fid'] + ['dist'] + ['dtm'] + [f'cond_{i+1}' for i in range(n_layers)]

    # Convert to DataFrame
    df = pd.DataFrame(np.array(results), columns=column_names)

    # Save to CSV
    if save_file:
        csv_filename = f"outputs/{line_no}_l2_conductivities.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n CSV saved to {csv_filename}")


#Run if script if executed
if __name__ == "__main__":
    print("Running L2 inversions with set up beta:")
    main(single_beta=84.40868089118096, conductivities_hsp = 0.03422309704591167)

    #first: 101.7424356306838
    #second: 84.40868089118096