# -*- coding: utf-8 -*-


# HMM with context layer
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import os, sys, glob, pdb, random, pickle
from scipy.special import softmax, expit
from pathlib import Path
import re


def get_paths():
	root = Path(os.getcwd())
	fig_path = root / 'recovery_250timest_blocks_oddball' 
	res_path = root / 'simulation_pickles_250timesteps' / 'Y_oddball'
	recovery_res_path = root / 'recovery_250timest_blocks_oddball'
	return fig_path,res_path,recovery_res_path


def save_figure(name): 
    fig_path,_,_ = get_paths()
    plt.savefig(fig_path / name)
    return
    

def psychosis_modulation(score, base_transition):
    """
    PPS weakens transitions between perceptual states. 
    in the most extreme case, the previous state tells us nothing about current state - uniform transitions!
    this should be the case with max schizophrenia
    here I just interpolate between uniform and "normal" base transition depending on the score
    """
    
    uniform_matrix = np.array([[0.5, 0.5], 
                               [0.5, 0.5]])
    
    score = pt.clip(score, 0, 1) # make sure score is in range 0, 1
    
    modulated_matrix = (1-score * 0.5) * base_transition + score * 0.5 * uniform_matrix

    assert pt.allclose(modulated_matrix.sum(axis=1), 1.0)
    
    modulated_matrix = modulated_matrix.reshape((1, 2, 2))
    
    return modulated_matrix

def autism_modulation(score, base_context): 
    """ 
    More ASD = less stability in transition matrix
    Less stability in the stable context.
    """
    
    adjusted_score = 1 - np.exp(-score)
    # update context based on score
    modulated_context = (1 - adjusted_score) * base_context + adjusted_score * np.array([[0.5, 0.5], [0.5, 0.5]])
    
    # assert that we have functioning prob dist
    assert pt.allclose(modulated_context.sum(axis=1), 1.0)
    
    modulated_context = modulated_context.reshape((1, 2, 2))
    
    return modulated_context
   

#timesteps = 50
#traces_timesteps = [i for i in all_traces if int(i.split('_')[-1].split('.')[0]) == timesteps]
#for trace in traces_timesteps: 


def compute_recovery(all_traces):

    blocks =5
    
    results_list = []
    
    for trace in all_traces:
        
        # traces einlesen
        filename = trace.name
        
        """
        Read prior predictive samples and extract simulated data from this
        """
        try: 
            with open(trace, 'rb') as handle: 
                full_trace = pickle.load(handle)
        except FileNotFoundError:
            print(f'File not found - skipping ### {filename} ###')
            continue
        
        # way to save generating AS and PS scores - not super fancy!
        # assuming this kind of structure: '500_timesteps_AS_0.0_PS_0.0_Y_rand.pickle'
        # I am using this structure at the moment: 'traces_AS_0.0_PS_0.0_Y_5change_Timest_15.pickle'
        # newest: traces_AS_1.0_PS_0.0_Y_stablechange_Timest_90_emissionmat_0.9.pickle
        #timesteps = float(filename.split('_')[0])
        #timesteps = 9 # DELETE LATER!
        timesteps = int(filename.split('Timest_')[1].split('.')[0])
        as_score = float(filename.split('_')[2])
        ps_score = float(filename.split('_')[4])
        data_name = filename.split('_PS_')[1].split('_Timest')[0].split('_', 1)[1]
        
        print('timesteps: ', timesteps, 
              'as_score: ', as_score, 
              'ps_score: ', ps_score,
              'data_name', data_name)
        
        # diese werte müssen mit denen aus der Simulation übereinstimmen. 
        num_timesteps = timesteps
        
        percepts_nodes      = [str(f'H_{i}') for i in range(num_timesteps)]
        observation_nodes   = [str(f'O_{i}') for i in range(num_timesteps)]
        context_nodes       = [str(f'C_{i}') for i in range(num_timesteps)]
    
        # get simulated data for hidden node (percepts) and observable nodes (stimuli) 
        trace = full_trace.posterior
                
        H_data = np.array([trace[node].values for node in percepts_nodes])
        H_data = H_data.reshape(H_data.shape[0], -1)
        # ALE: shape (50, 2400) means ~50 trials, 2400 blocks
    
        # I used the actually observed data from the sims here
        O_data = np.array([full_trace.observed_data[node].values for node in observation_nodes])
        O_data = O_data.flatten()
        O_data = O_data.reshape(-1, 1).repeat(H_data.shape[1], axis=1)
    
        C_data = np.array([trace[node].values for node in context_nodes])
        C_data = C_data.reshape(C_data.shape[0], -1)
    
        
        """
        Set up model architecture and probability distributions
        """
    
        # hidden states are categorical dist with transition matrix
        n_context_states = 2 # stable, variable
        n_hidden_states = 2 # left, right
        n_observable_states = 2 # left, right
    
        # context transitions. how to read these arrays: 
        # ROWS: t-1 state (previous state), COLS: t state (next) 
        # e.g. row 1, col 1: stable - stable, row 1, col 2: stable - variable
        context_matrix =    np.array([[0.8, 0.2],
                                      [0.3, 0.7]]) 
    
        # transition matrix in the percept layer depends on the context
        transition_matrix_given_context = {
            0: np.array([[0.9, 0.1],  # stable context. left-left: 0.8, left-right:0.2
                         [0.1, 0.9]]), # right-left: 0.2, right-right:0.8
            1: np.array([[0.5, 0.5],  # variable context
                         [0.5, 0.5]])
        }
        
            # emission matrix
        emission_matrix = np.array([[0.75, 0.25],   # 0.9: observing left when in perceptual state left
                                    [0.25, 0.75]])  # 0.1: observing right when in perceptual state left
        
        # prior on context
        context_priors = np.array([0.5, 0.5]) # slight preference for stable
    
        # prior on percepts
        percept_priors = np.array([0.5, 0.5]) # no bias for any one stimulus type
        
        with pm.Model() as hmm_model:
            
            
            asd_score = pm.Beta('ASD_score', alpha=1, beta=1)
            ssd_score = pm.Beta('SSD_score', alpha=1, beta=1)
            constant_param = 'none_constant'
            
            # first context 
            context_states = [pm.Categorical('C_0', p=context_priors, observed=C_data[0,:blocks])] #:10 means: observe first 10 blocks
            
            # first hidden state - observe response
            hidden_states = [pm.Categorical('H_0', p=percept_priors, observed=H_data[0,:blocks])]
            
            # list for observable states - observe stimulus
            observable_states = [pm.Categorical('O_0', p=pm.math.switch(pt.eq(hidden_states[0].reshape((-1, 1)), 0), emission_matrix[0].reshape(1, -1), 
                                                                        emission_matrix[1].reshape(1, -1)), observed=O_data[0,:blocks])]
            
            for t in range(1, num_timesteps):
                
               # 1. TOP LAYER: context
               # ASD modulation: Increased ASD score reduces stability of stable context
               asd_scored_context_matrix = autism_modulation(asd_score, context_matrix)
               
               context_transition_probs = pm.math.switch(
                   pt.eq(context_states[t-1].reshape((-1, 1)), 0),
                   asd_scored_context_matrix[:,0,:],
                   asd_scored_context_matrix[:,1,:]
               )
                                                         
               context_state_t = pm.Categorical(f'C_{t}', p=context_transition_probs, observed=C_data[t, :blocks])
               
               context_states.append(context_state_t)
               
               # 2. MID LAYER: percepts           
               # PPS Modulation: Increased PPS score makes perceptual transitions less stable
               transition_matrix_t = pm.math.switch(
                   pt.eq(context_state_t.reshape((-1, 1, 1)), 0),
                   psychosis_modulation(ssd_score, transition_matrix_given_context[0]),
                   psychosis_modulation(ssd_score, transition_matrix_given_context[1])
                )
               
               # Perceptual state at time t
               hidden_state_t = pm.Categorical(f'H_{t}', 
                                              p=pm.math.switch(pt.eq(hidden_states[t-1].reshape((-1, 1)), 0), 
                                                               transition_matrix_t[:,0,:], 
                                                               transition_matrix_t[:,1,:]), 
                                              observed=H_data[t, :blocks])
              
               hidden_states.append(hidden_state_t)
              
               # 3. BOTTOM LAYER: observations
               # Observable state at time t depending on hidden state
               observable_state_t = pm.Categorical(f'O_{t}', 
                                                  p=pm.math.switch(pt.eq(hidden_state_t.reshape((-1, 1)), 0), emission_matrix[0].reshape(1, -1), 
                                                                   emission_matrix[1].reshape((1, -1))),
                                                  observed=O_data[t, :blocks])
              
               observable_states.append(observable_state_t)
    
        
        with hmm_model:
            
            # option to explicitly set categorical Gibbs metropolis Sampler bc of errors previously
            # pm.CategoricalGibbsMetropolis(context_states, hidden_states)
            
            csv_filename = f'blocks_{blocks}_recovery_AS_{as_score}_PS_{ps_score}_{data_name}_T_{timesteps}_const_{constant_param}.csv'
            pickle_filename = f'blocks_{blocks}_recovery_AS_{as_score}_PS_{ps_score}_{data_name}_T_{timesteps}_const_{constant_param}.pickle'

            _,_,recovery_res_path=get_paths()
            file_path = recovery_res_path / csv_filename
            
            if file_path.exists(): 
                print('File exists: ', csv_filename)
                
                continue
                
            else: 
                print('Proceed with model: ', csv_filename)
            
                trace = pm.sample(500, tune=250, target_accept=0.9, cores=1, chains=4, return_inferencedata=True) 
            
                with open(recovery_res_path / pickle_filename, 'wb') as handle: 
                    pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                summary = az.summary(trace, var_names=['ASD_score', 'SSD_score'])
                print(summary)
                
                # plot posterior over scores        
                az.plot_posterior(trace, var_names=['ASD_score', 'SSD_score'], figsize=(10,5))
                save_figure(f'AS_{as_score}_PS_{ps_score}_{data_name}_timest_{num_timesteps}_posterior.svg')
                
                
                # Summarize the trace for ASD_score and SSD_score
                summary = az.summary(trace, var_names=['ASD_score', 'SSD_score'])
            
                # Extract the desired values for ASD_score and SSD_score
                ASD_mean = summary.loc['ASD_score', 'mean']
                ASD_sd = summary.loc['ASD_score', 'sd']
                ASD_hdi_3 = summary.loc['ASD_score', 'hdi_3%']
                ASD_hdi_97 = summary.loc['ASD_score', 'hdi_97%']
            
                SSD_mean = summary.loc['SSD_score', 'mean']
                SSD_sd = summary.loc['SSD_score', 'sd']
                SSD_hdi_3 = summary.loc['SSD_score', 'hdi_3%']
                SSD_hdi_97 = summary.loc['SSD_score', 'hdi_97%']
                
                results_model = {
                    'Timesteps': num_timesteps,
                    'Data': data_name,
                    'as_score_gt': as_score,
                    'ASD_score_recovered': ASD_mean,
                    'ASD_sd': ASD_sd,
                    'ASD_hdi_3%': ASD_hdi_3,
                    'ASD_hdi_97%': ASD_hdi_97,
                    'ps_score_gt': ps_score,
                    'SSD_score_recovered': SSD_mean,
                    'SSD_sd': SSD_sd,
                    'SSD_hdi_3%': SSD_hdi_3,
                    'SSD_hdi_97%': SSD_hdi_97,
                    'blocks': blocks
                }
                
                
                temp_df = pd.DataFrame(results_model, index=[0])
                temp_df.to_csv(recovery_res_path / csv_filename)
            
                # Append results to the list
                results_list.append(results_model)
                
        df = pd.DataFrame.from_dict(results_list)
        df.to_csv(recovery_res_path / f'recovery_results_blocks_{blocks}_Timest_{timesteps}.csv')

    return df

