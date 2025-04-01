# -*- coding: utf-8 -*-


import pytensor
pytensor.config.optimizer = 'fast_run'

# HMM with context layer
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import os, sys, glob, pdb, random, pickle
from scipy.special import softmax
import subprocess
import random
#import psutil
import time
import threading
import multiprocessing

"""
HMM with context layer. 

- Observation, perception and context layers. 
- transition between percepts depends on current context state
"""

root = os.getcwd()
res_path = os.path.join(root, 'master_loop_results')

def psychosis_modulation(score, base_transition):
    """
    PPS weakens transitions between perceptual states. 
    in the most extreme case, the previous state tells us nothing about current state - uniform transitions!
    this should be the case with max schizophrenia
    here I just interpolate between uniform and "normal" base transition depending on the score
    """
    
    if np.isclose(score, 0.0): 
        return base_transition 
    
    uniform_matrix = np.array([[0.5, 0.5], 
                               [0.5, 0.5]])
    
    score = np.clip(score, 0, 1) # make sure score is in range 0, 1
    
    modulated_matrix = (1-score * 0.5) * base_transition + score * 0.5 * uniform_matrix

    assert np.all(np.isclose(modulated_matrix.sum(axis=1), 1.0))
    
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
    assert np.all(np.isclose(modulated_context.sum(axis=1), 1.0))
    
    return modulated_context

def make_change_observations(num_switches):
    """
    make observation data based on number of timesteps and switches
    """
    array = np.zeros(num_timesteps)
    switch_points = sorted(np.random.choice(range(1, num_timesteps), num_switches, replace=False))
    
    current_value = 0
    previous_switch = 0
    
    for switch in switch_points:
        array[previous_switch:switch] = current_value
        
        current_value = 1 - current_value
        
        previous_switch = switch
    
    array[previous_switch:] = current_value
    return array

def generate_stable_then_alternating(num_timesteps, seed=None):
    """
    Generate a sequence of alternating "stable" and "alternating" blocks.
    The first block is randomly chosen to be either stable or alternating.
    Stable blocks hold either 0 or 1, alternating blocks flip between 0 and 1.
    """
    if seed is not None: 
        random.seed(seed)
    
    result = np.zeros(num_timesteps, dtype=float)  # Start with an array of zeros
    i = 0  # Initialize index
    
    # Randomly decide if the first block will be stable or alternating
    next_block_type = random.choice(['stable', 'alternating'])
    
    while i < num_timesteps:
        # Randomly choose segment length
        segment_length = random.randint(5, 20) # set to fix number
        
        # Trim the segment length if it exceeds remaining timesteps
        segment_length = min(segment_length, num_timesteps - i)
        
        if next_block_type == 'stable':
            stable_value = random.choice([0.0, 1.0])
            result[i:i + segment_length] = stable_value
            next_block_type = 'alternating'
        
        elif next_block_type == 'alternating':
            # Generate alternating 0 and 1 for the length of the segment
            alternating_values = np.array([1.0 if j % 2 == 0 else 0.0 for j in range(segment_length)])
            result[i:i + segment_length] = alternating_values
            next_block_type = 'stable'
        
        i += segment_length
    
    return result


def generate_stable_then_random(num_timesteps, seed=None): 
    
    """
    generate alternating stable vs random data
    """
    
    if seed is not None: 
        random.seed(seed)
    
    result = np.zeros(num_timesteps, dtype=float)  # Start with an array of zeros
    i = 0  # Initialize index
    
    # Randomly decide if the first block will be stable or alternating
    next_block_type = random.choice(['stable', 'random'])
    
    while i < num_timesteps:
        # Randomly choose segment length
        segment_length = random.randint(5, 20) # set to fix number
        
        # Trim the segment length if it exceeds remaining timesteps
        segment_length = min(segment_length, num_timesteps - i)
        
        if next_block_type == 'stable':
            stable_value = random.choice([0.0, 1.0])
            result[i:i + segment_length] = stable_value
            next_block_type = 'random'
        
        elif next_block_type == 'random':
            # Generate alternating 0 and 1 for the length of the segment
            random_values = np.array([random.choice([0,1]) for i in range(segment_length)])
            result[i:i + segment_length] = random_values
            next_block_type = 'stable'
        
        i += segment_length
    
    return result

def generate_contextdependent_probabilisticswitching(num_timesteps):
    """
    In stable context the probabilities are 80% that the percept is the one as before, in variable it's 50%'
    """
    random.seed(123)  # Set fixed random seed for repeatability
    
    result = np.zeros(num_timesteps, dtype=float)  # Start with an array of zeros
    i = 0  # Initialize index
    
    # Randomly decide if the first block will be stable or variable
    next_block_type = random.choice(['stable', 'variable']) # could set it to randomly start either with stable or variable
    next_block_type = 'stable'
    
    while i < num_timesteps:
        segment_length = random.randint(5, 20) # could give it random segment length
        #segment_length = 10
        
        # Trim the segment length if it exceeds remaining timesteps
        segment_length = min(segment_length, num_timesteps - i)
        
        if next_block_type == 'stable':
            current_value = random.choice([0, 1]) if i == 0 else result[i - 1]
            for j in range(segment_length):
                
                if random.random() < 0.9: # 90% chance of keeping the same value, 10% of switching
                    result[i + j] = current_value
                else:
                    current_value = 1 - current_value  # Flip value
                    result[i + j] = current_value
            next_block_type = 'variable'
        
        elif next_block_type == 'variable':
            current_value = random.choice([0, 1]) if i == 0 else result[i - 1]
            for j in range(segment_length):
                
                if random.random() < 0.66:# 50% chance of keeping the same value, 50% of switching
                    result[i + j] = current_value
                else:
                    current_value = 1 - current_value  # Flip value
                    result[i + j] = current_value
            next_block_type = 'stable'
        
        i += segment_length
    
    return result

def generate_oddballinstablecontext(num_timesteps):
    """
    In stable context, the probabilities are 90% that the percept is 0 or 1 independently of the previous one.
    In variable context, there is a 50% chance of keeping the same value or switching.
    """
    random.seed(123)  # Set fixed random seed for repeatability
    
    result = np.zeros(num_timesteps, dtype=float)  # Start with an array of zeros
    i = 0  # Initialize index
    
    # Randomly decide if the first block will be stable or variable
    next_block_type = random.choice(['stable', 'variable']) # could set it to randomly start either with stable or variable
    next_block_type = 'stable'
    
    last_stable_value = 0  # To alternate between 0 and 1 in stable context
    
    while i < num_timesteps:
        segment_length = random.randint(5, 20)  # Random segment length
        
        # Trim the segment length if it exceeds remaining timesteps
        segment_length = min(segment_length, num_timesteps - i)
        
        if next_block_type == 'stable':
            # In stable context, alternate starting value between 0 and 1 with 90% chance of keeping
            current_value = last_stable_value
            
            for j in range(segment_length):
                if random.random() < 0.9:  # 90% chance to keep the value
                    result[i + j] = current_value
                else:
                    # Flip the value with a 10% chance
                    current_value = 1 - current_value
                    result[i + j] = current_value
            
            # Alternate the starting value for the next stable block
            last_stable_value = 1 - current_value
            next_block_type = 'variable'
        
        elif next_block_type == 'variable':
            # In variable context, 50% chance of keeping the same value or switching
            current_value = random.choice([0, 1]) if i == 0 else result[i - 1]
            for j in range(segment_length):
                if random.random() < 0.5:  # 50% chance to keep the value
                    result[i + j] = current_value
                else:
                    # Flip the value with 50% chance
                    current_value = 1 - current_value
                    result[i + j] = current_value
            next_block_type = 'stable'
        
        i += segment_length
    
    return result
           

    


def extract_posteriors(variable_type='H'):
    """
    take posteriors from sampling for plotting... 
    """
    var_names = [f"{variable_type}_{i}" for i in range(0,num_timesteps)]
    sampled_results = [trace.posterior[i].values for i in var_names]
    results_concat = [np.concatenate(sampled_results[t], axis=0) for t in range(num_timesteps)]
    
    posteriors = []
    
    for t in range(num_timesteps): 
        state_counts = np.bincount(results_concat[t], minlength=2)
        total_samples = len(results_concat[t])
        posterior_prob_t = state_counts / total_samples
        posteriors.append(posterior_prob_t)
        
    return posteriors


#def monitor_cpu_usage(interval=1):
#    while sampling_in_progress:
#        print(f"CPU usage: {psutil.cpu_percent(percpu=True)}")
#        time.sleep(interval)
        
def run_modulation_model_parallel(par): 
    
    num_timesteps, data, data_name, as_score, ps_score = par
    return run_modulation_model(num_timesteps, data, data_name, as_score, ps_score)

def run_modulation_model(num_timesteps, data, data_name, as_score=0.0, ps_score=0.0): 
    """
    run the whole model with args: as_score, ps_score
    """
    
    with pm.Model() as hmm_model:
        
        # first context 
        context_states = [pm.Categorical('C_0', p=context_priors)]
        
        # first hidden state
        hidden_states = [pm.Categorical('H_0', p=percept_priors)]
        
        # list for observable states
        observable_states = [pm.Categorical('O_0', p=pm.math.switch(pt.eq(hidden_states[0], 0), emission_matrix[0], emission_matrix[1]), observed=data[0])]
        
        consecutive_same_obs = 0
        
        # context transition modulated by ASD
        asd_modulated_matrix = autism_modulation(as_score, context_matrix)
        
        # percept transition modulated by SSD
        pps_modulated_matrix_stable = psychosis_modulation(ps_score, transition_matrix_given_context[0])
        pps_modulated_matrix_variab = psychosis_modulation(ps_score, transition_matrix_given_context[1])
        
        trans_matrix_t_modulated = {0: pps_modulated_matrix_stable, 
                                    1: pps_modulated_matrix_variab}
        
        for t in range(1, num_timesteps):
            
            consecutive_same_obs += int(data[t] == data[t-1])
            
            context_transition_probs = pm.math.switch(
                pt.eq(context_states[t-1], 0),
                asd_modulated_matrix[0],
                asd_modulated_matrix[1])
            
    
            #context_transition_probs = pm.math.switch(pt.eq(context_states[t-1], 0),
            #                                          context_matrix[0],
            #                                          context_matrix[1])
                                                      
            context_state_t = pm.Categorical(f'C_{t}', p=context_transition_probs)
            
            context_states.append(context_state_t)
           
           
            # Transition probabilities for hidden states depend on current context state
            transition_matrix_t = pm.math.switch(pt.eq(context_state_t, 0),
                                                trans_matrix_t_modulated[0],
                                                trans_matrix_t_modulated[1])
           
            
            # Perceptual state at time t
            hidden_state_t = pm.Categorical(f'H_{t}', 
                                           p=pm.math.switch(pt.eq(hidden_states[t-1], 0), 
                                                            transition_matrix_t[0], 
                                                            transition_matrix_t[1]))
           
            hidden_states.append(hidden_state_t)
           
            # Observable state at time t depending on hidden state
            observable_state_t = pm.Categorical(f'O_{t}', 
                                               p=pm.math.switch(pt.eq(hidden_state_t, 0), emission_matrix[0], emission_matrix[1]),
                                               observed=data[t])
           
            observable_states.append(observable_state_t)
            
    with hmm_model:
            
            # for marc3
            #num_procs = int(args.numprocs)
            num_procs = 8
            
            os.chdir(res_path)
            filename = os.path.join(f'traces_AS_{as_score}_PS_{ps_score}_{data_name}_Timest_{num_timesteps}.pickle')
            
            file_path = os.path.join(res_path, filename)
            
            if os.path.exists(file_path): 
                print('File exists: ', filename)
                
                return 
                
            else: 
                print('Proceed with model: ', filename)
            
                trace = pm.sample(num_samples, tune=num_tune_in, cores=num_procs, chains=8, return_inferencedata=True)  
                
                # check cpu usage
                #for _ in range(10):
                    #print(f"CPU usage: {psutil.cpu_percent(percpu=True)}")
                    #time.sleep(1)
            
                with open(filename, 'wb') as handle: 
                    pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
                return trace
        

if __name__ == '__main__':
    
    # BUILD MODEL ~.+.+.+.+.+.+.+.+.+.+.+.

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
    emission_matrix = np.array([[0.75, 0.25],   # 0.8: observing left when in perceptual state left
                                [0.25, 0.75]])  # 0.2: observing right when in perceptual state left
    # prior on context
    context_priors = np.array([0.5, 0.5]) # no prior bias for any context

    # prior on percepts
    percept_priors = np.array([0.5, 0.5]) # no bias for any one stimulus type
    
    # some important global variables.
    num_samples = 500
    num_tune_in = 100
    
    # num_timesteps 50 - 450
    # shape of data [maybe 5 different options or so]
    # verschiedene AS/ PS scores [10 combinations]
    # (observations)
    min_timesteps = 50 
    max_timesteps = 150 # used to be 450
    
    all_num_timesteps = np.arange(min_timesteps , max_timesteps+1 , 50)
    
    all_subjects = [{'as_score': 0.1, 'ps_score': 0.9}, 
                    {'as_score': 0.9, 'ps_score': 0.1},
                    {'as_score': 0.5, 'ps_score': 0.5},
                    {'as_score': 0.1, 'ps_score': 0.1},
                    {'as_score': 0.9, 'ps_score': 0.9}
                    ]

    # insert here the timesteps to loop over
    for num_timesteps in sorted(all_num_timesteps, reverse=True):

        
        print(f'Starting with timesteps {num_timesteps}')
        
        Y_stablechange = generate_stable_then_alternating(num_timesteps, seed=123)
        Y_stablerandom = generate_stable_then_random(num_timesteps, seed=123)
        num_switches = int(num_timesteps / 10)
        Y_fixedchange  = make_change_observations(num_switches)
        Y_newprob = generate_contextdependent_probabilisticswitching(num_timesteps)
        Y_oddball = generate_oddballinstablecontext(num_timesteps)
        
        Y_singlechange = [0,0,0,0,0,0,0,1,1,1]
        Y_oddballtest = [0,0,0,0,0,0,0,1,0,0]
        Y_stablealternating = [0,0,0,0,0,0,0,1,0,1]
        Y_semioddball = [0,0,0,0,0,0,0,1,1,0]
        
        
        
        
        all_data = [{'Y_stablechange': Y_stablechange},
                    {'Y_stablerandom': Y_stablerandom}, 
                    {'Y_fixedchange': Y_fixedchange},
                    #{'Y_newprob': Y_newprob},
                    {'Y_oddball': Y_oddball}
                    #{'Y_singlechange': Y_singlechange},
                    #{'Y_oddballtest': Y_oddballtest},
                    #{'Y_stablealternating': Y_stablealternating},
                    #{'Y_semioddball': Y_semioddball}
                    ]        
        
        
        params=[]
        for data_dict in all_data:
            
            data_name = list(data_dict.keys())[0]
            data = data_dict[data_name]
            
            print(f'Data type: {data_name}')
        
            for subject in all_subjects:
                
                as_score = subject['as_score']
                ps_score = subject['ps_score']
                
                print(f'Subject: AS: {as_score}, PS: {ps_score}')
                
                params.append((num_timesteps, data, data_name, as_score, ps_score))
                
        # multiple datasets in parallel        
        #process_pool = multiprocessing.Pool(processes=5)
                
        #for trace in process_pool.imap(run_modulation_model_parallel,params):
        
        for param in params:
            num_timesteps, data, data_name, as_score, ps_score = param

            trace = run_modulation_model(num_timesteps, data, data_name, as_score, ps_score)
            
            
            if trace is not None:
    
                posterior_percepts = extract_posteriors(variable_type='H')
                posterior_contexts = extract_posteriors(variable_type='C')
                
                # plot this now
                timesteps = np.arange(num_timesteps)
                posteriors_percept_0 = [p[0] for p in posterior_percepts]
                posteriors_percept_1 = [p[1] for p in posterior_percepts]
                
                posteriors_context_0 = [p[0] for p in posterior_contexts]
                posteriors_context_1 = [p[1] for p in posterior_contexts]
                
                plt.figure()
                plt.plot(timesteps, posteriors_percept_0, label='Percept Left')
                plt.plot(timesteps, posteriors_percept_1, label='Percept Right')
                plt.plot(timesteps, posteriors_context_0, label='stable context', color='darkblue', linestyle='--')
                plt.plot(timesteps, posteriors_context_1, label='variable context', color='coral', linestyle='--')
                plt.xlabel('Timestep')
                plt.ylabel('Posterior Probability')
                plt.legend()
                plt.title('Posterior Probabilities for Hidden States Over Time')
                plt.savefig(f'plot_AS_{as_score}_PS_{ps_score}_timesteps_{num_timesteps}_data_{data_name}.svg')
        
