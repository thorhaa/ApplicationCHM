# -*- coding: utf-8 -*-


import os, glob
import pandas as pd
import multiprocessing

from modulation_recovery_Thorben import compute_recovery,get_paths

#for timesteps in range(50, 451, 50):
    #print(f"Starting with {timesteps} timesteps")
timesteps = 250

if __name__=="__main__":
     
    multiprocessing.set_start_method("spawn",force=True)
    
    _,result_path,_ = get_paths()
            
    all_traces = result_path.glob('*.pickle')
        
    traces_timesteps = list(filter(lambda i:int(str(i).split('_')[-1].split('.')[0]) == timesteps,all_traces))
    
    # optional args: asd or ssd constant - default: both are free params.
    result_df     = compute_recovery(traces_timesteps)
    #result_df_asd = compute_recovery(traces_timesteps, asd_constant=True, ssd_constant=False)
    #result_df_ssd = compute_recovery(traces_timesteps, asd_constant=False, ssd_constant=True)

