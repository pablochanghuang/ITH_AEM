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


### Setting up frequencies ###

##Creating a list of all frequencies string:
all_freqs = ['140k', '40k', '8200', '3300', '1800', '400'] ## All frequencies
freq_cx_str = ['3300'] ## Coaxial frequencies - string array
all_freqs = np.array([f for f in all_freqs if f not in freq_cx_str]) ## Removes the coaxial frequencies from the list

## Create numpy array with freqs:
all_freqs_num = np.r_[135, 40, 8.2, 3.3, 1.8, 0.4]*1e3 ## Creates a numpy array with all frequencies
all_freqs_num[::-1].sort()

# Delete frequencies, coaxial freqs
freq_cx = [3300] ## Coaxial frequencies - numpy array
all_freqs_num = np.delete(all_freqs_num, np.where(all_freqs_num == freq_cx)) ## Removes the coaxial frequencies from the list

def survey_object(line_station, freq_avoid, rm_real =True, rm_imag = True):

    ## Source orientation
    tx_orientation = ["z", "z", "z", "z", "z"] # "x", "y" or "z" - here we are using only the CP frequencies
    moments = [17, 49, 72, 187, 359]  ## Tx dipole moments fromm Xcalibur´s report -  only cp geometry

    ## Receiver orientation
    rx_orientation = ["z", "z", "z", "z", "z"]  # "x", "y" or "z" - here we are using only the CP frequencies
    dtype = 'ppm'  # "secondary", "total" or "ppm"

    #### Generate Source and Receivers locations ####

    # Generate the labels with frequency values strings
    source_locations = np.array([[
                getattr(line_station, f'cp{freq}_tx_x'),  # Transmitter X
                getattr(line_station, f'cp{freq}_tx_y'),  # Transmitter Y
                getattr(line_station, 'height_tx')  # Common height for the transmitter
                ] for freq in all_freqs])

    receiver_locations = np.array([[
                getattr(line_station, f'cp{freq}_rx_x'),  # Transmitter X
                getattr(line_station, f'cp{freq}_rx_y'),  # Transmitter Y
                getattr(line_station, 'height_tx')  # Common height for the transmitter
                ] for freq in all_freqs])

    # Create soyrce and receiver lists:

    #Set up frequencies to avoid if needed - GET RID of Higher freqs
    freqs = all_freqs_num
    #freqs_names = all_freqs

    #### Create source and receiver list ####
    source_list = []

    for i, freq in enumerate(freqs):
    
        receiver_list = []

        #Add real readings
        if rm_real:
            if freq not in freq_avoid:
                receiver_list.append(
                    fdem.receivers.PointMagneticFieldSecondary(
                        locations=np.c_[receiver_locations[i]].T,
                        orientation=rx_orientation[i],
                        data_type=dtype,
                        component="real",
                    )
                )
        else:
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    locations=np.c_[receiver_locations[i]].T,
                    orientation=rx_orientation[i],
                    data_type=dtype,
                    component="real",
                )
            )
        if rm_imag:
                if freq not in freq_avoid:
                    receiver_list.append(
                        fdem.receivers.PointMagneticFieldSecondary(
                            locations=np.c_[receiver_locations[i]].T,
                            orientation=rx_orientation[i],
                            data_type=dtype,
                            component="imag",
                        )
                    )
        else:
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    locations=np.c_[receiver_locations[i]].T,
                    orientation=rx_orientation[i],
                    data_type=dtype,
                    component="imag",
                )
            )

        #Apprend source
        source_list.append(
                fdem.sources.MagDipole(
                    receiver_list=receiver_list,
                    frequency=freq,
                    location=np.c_[source_locations[i]].T,
                    orientation=tx_orientation[i],
                    moment=moments[i],
                )
            )

    #### Define the survey object ####

    survey = fdem.survey.Survey(source_list)
    #survey.nD ## Check the number of Tx and Rx components (Real and Imaginary)

    return survey


def data_object(line_station, relative_error, noise_floor, survey, freq_avoid, freqs_names = all_freqs, rm_real =True, rm_imag = True):
    
    #Get arrays of data from sounding, real and img
    inphase_data = []
    quad_data = []

    for f in freqs_names:
        inphase_data.append(line_station[f'cpi{f}'])
        quad_data.append(line_station[f'cpq{f}'])

    inphase_data = np.array(inphase_data)
    quad_data = np.array(quad_data)


    #### get rid of frequencies to avoid ####
    if rm_real and rm_imag:
        ind_avoid = [np.where(all_freqs_num == f)[0][0] for f in freq_avoid]
    elif rm_real:
        ind_avoid = [np.where(all_freqs_num == f)[0][0] for f in freq_avoid]
    elif rm_imag:
        ind_avoid = [np.where(all_freqs_num == f)[0][0] for f in freq_avoid]
    else:
        ind_avoid = [None]
    
    #Put everything in a single array

    dobs = []
    for i in range(len(inphase_data)):

        # if i == 3: ## In this analysis, we are not inverting the coaxial data
        #     pass
        # else:
        if rm_real and not rm_imag:
            if i not in ind_avoid:
                dobs = np.append(dobs, inphase_data[i])
            dobs = np.append(dobs, quad_data[i])
        elif rm_imag and not rm_real:
            dobs = np.append(dobs, inphase_data[i])
            if i not in ind_avoid:
                dobs = np.append(dobs, quad_data[i])
        elif rm_real and rm_imag:
            if i not in ind_avoid:
                dobs = np.append(dobs, inphase_data[i])
                dobs = np.append(dobs, quad_data[i])
        else:
            dobs = np.append(dobs, inphase_data[i])
            dobs = np.append(dobs, quad_data[i])

    #### Create data object ####

    # Defines the data object and assigns uncertainties with the noise floor
    data_object = data.Data(survey, dobs=dobs, relative_error=relative_error, noise_floor=noise_floor)

    return data_object

def mapping_forward_objects(layer_thicknesses = [], n_layers = 1, survey = None, depth_max = None, depth_min = None, geometric_factor = None):

    #### Define model mesh ####
    if n_layers > 1:
        #Set up layers for L2 inversion:
        ### Depth max ### ~ From skin depth

        ### Mesh Definition ###

        #Increase layer thickness by geometric factor until depth_max
        layer_thicknesses = [depth_min]
        while np.sum(layer_thicknesses) < depth_max:
            layer_thicknesses.append(geometric_factor*layer_thicknesses[-1])

        #Set number of layers
        n_layers = len(layer_thicknesses) + 1  # Number of layers

        #set for Simulations
        thicknesses = layer_thicknesses

        h = np.r_[layer_thicknesses, layer_thicknesses[-1]]

    
    else:
        thicknesses = []

        h = np.r_[layer_thicknesses]


    #### Define the mapping ####
    log_mapping = maps.ExpMap(nP=n_layers)

    #### Define Simulation Object (physics, forward simulation)####

    simulation = fdem.Simulation1DLayered(
        survey=survey,
        thicknesses= thicknesses,
        sigmaMap=log_mapping
    )

    #### Create regularization mesh ####
    regularization_mesh = TensorMesh([h], "N")

    return regularization_mesh, log_mapping, simulation, n_layers

def regularization_object(regularization_mesh, reference_conductivity_model, alpha_s, alpha_x, sparse_regularization=False, norms=[0,0]):

    if sparse_regularization:
        #Set up Sparse regularization
        # Define regularization
        reg = regularization.Sparse(
                regularization_mesh,
                length_scale_x=10.0,
                reference_model=reference_conductivity_model,
                reference_model_in_smooth=False
            )
        #Define sparse and blocky norms

        # norms[0] controls sparsity in the model values.
        # norms[1] controls sparsity in the model gradients (i.e., how blocky vs. smooth the changes are).
        reg.norms = norms
    
    else:
        #Set up L2 and halfspace regularization
        reg = regularization.WeightedLeastSquares(
        regularization_mesh,
        length_scale_x=10.0,
        reference_model=reference_conductivity_model,
        reference_model_in_smooth=False
        )

    #Set regularization parameters:
    reg.alpha_s = alpha_s  #Model to reference model
    reg.alpha_x= alpha_x #Controls roughness penalty

    return reg

def inversion_setup(dmis, reg, opt, single_beta = None, beta_ratio  = 1e1, sparse_inversion=False, IRLS_coolrate=2, IRLS_coolfactor=1.5 , IRLS_percentile=100, inv_prob_return = False, save_outputs=False):

    if sparse_inversion:
        #Check if single beta for sparse inversion
        if single_beta is not None:
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta = single_beta)

            #Inversion Directives
            update_jacobi = directives.UpdatePreconditioner() #Helps stabilize each IRLS step.

            # Directive for the IRLS 
            update_IRLS = directives.UpdateIRLS(cooling_rate = IRLS_coolrate, # update weights every x iterations
                                                cooling_factor=IRLS_coolfactor, # stronger update to weights
                                                percentile = IRLS_percentile, #controls how much of the model is affected by reweighting.
                                                max_irls_iterations=50)

            save_output = directives.SaveOutputDictEveryIteration()

            directives_list = [
                update_IRLS, 
                update_jacobi,
                save_output
            ]
            print("SINGLE BETA")
        
        else: 
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

            #Inversion Directives
            update_jacobi = directives.UpdatePreconditioner() #Helps stabilize each IRLS step.
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=beta_ratio)

            # Directive for the IRLS 
            update_IRLS = directives.UpdateIRLS(cooling_rate = IRLS_coolrate, # update weights every x iterations
                                                cooling_factor=IRLS_coolfactor, # stronger update to weights
                                                percentile = IRLS_percentile, #controls how much of the model is affected by reweighting.
                                                max_irls_iterations=50)
            
            save_output = directives.SaveOutputDictEveryIteration()

            directives_list = [
                update_IRLS, 
                update_jacobi,
                starting_beta,
                save_output
            ]

    
    elif single_beta is not None:

        #Set inversion without beta cooling, single beta value
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta = single_beta)

        #Set inversion directives:
        update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
        #starting_beta = directives.BetaEstimate_ByEig(beta0_ratio = beta_ratio)
        #beta_schedule = directives.BetaSchedule(coolingFactor=1.5, coolingRate=2)
        target_misfit = directives.TargetMisfit(chifact=1.0)
        save_output = directives.SaveOutputDictEveryIteration()

        directives_list = [
            update_jacobi,
            #starting_beta,
            #beta_schedule,
            target_misfit,
            save_output
        ]

    else:
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Set inversion directives:
        update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio = beta_ratio)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.5, coolingRate=2)
        target_misfit = directives.TargetMisfit(chifact=1.0)
        save_output = directives.SaveOutputDictEveryIteration()

        directives_list = [
            save_output, 
            update_jacobi,
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

    #### Combine the inverse problem and the set of directives ####
    inv = inversion.BaseInversion(inv_prob, directives_list)
    
    # If want to return also inv_prob

    
    if inv_prob_return:
        if save_outputs:
            return inv, inv_prob, save_output
        else:
            return inv, inv_prob
    else:
        if save_outputs:
            return inv, save_output
        return inv




            
        