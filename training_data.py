import random
import numpy as np
import pandas as pd
import torch
from mpfd_solver import setup, ModelRun  # Assuming these functions are in your script
import os


def generateData(num_samples):
    dt = 1  # Time-step
    zN = 40  # Domain size
    N = 399  # Number of gridpoints
    
    file_path = 'high_fidelity_training_data.csv'
    file_exists = os.path.isfile(file_path)
    
    # Open the CSV file in append mode
    with open(file_path, 'a') as f:
        # Write headers if the file does not exist
        if not file_exists:
            f.write('tN,psiInitial,psiFinal\n')
        
        for _ in range(num_samples):
            tN = random.uniform(360, 1800)
            psiB = random.uniform(-150, -50)
            psiT = random.uniform(-50, -10)
            
            psiInitial = sorted([random.uniform(psiB, psiB+1) for _ in range(N)])
            psiInitial = np.hstack((psiB, psiInitial, psiT))
            
            z, t, dz, n, nt, zN, psi_list, psiB, psiT, pars = setup(dt, tN, zN, psiInitial)
            output, err = ModelRun(dt, dz, n, nt, psi_list, psiB, psiT, pars)
            
            psiFinal = np.hstack((psiB, output[-1], psiT))
            
            # Prepare the data for saving
            data = {
                'tN': tN,
                'psiInitial': psiInitial.tolist(),
                'psiFinal': psiFinal.tolist()
            }
            
            # Convert data to DataFrame to save as CSV
            df = pd.DataFrame([data])
            
            # Append the data to the CSV file
            df.to_csv(f, header=False, index=False)
    
    print("Data generation completed and saved to 'high_fidelity_training_data.csv'")

