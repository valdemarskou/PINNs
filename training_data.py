import random
import numpy as np
import pandas as pd
import torch
import torch_mpfd_solver as torchsolver
import os


def generateData(num_samples):
    dt = 10  # Time-step
    zN = 40  # Domain size
    
    file_path = 'high_fidelity_training_data.csv'
    file_exists = os.path.isfile(file_path)
    
    # Open the CSV file in append mode
    with open(file_path, 'a') as f:
        # Write headers if the file does not exist
        if not file_exists:
            f.write('t,output\n')
        
        for _ in range(num_samples):
            tN = random.uniform(360, 1440)
            psiB = random.uniform(-150, -50)
            psiT = random.uniform(-50, -30)

            flag = 0
            
            psiInitial = sorted([random.uniform(psiB,psiB) for _ in range(int(zN-1))])
            psiInitial = np.hstack([psiB,psiInitial,psiT])
            
            z,t,dts,dz,n,nt,zN,psi,psiB,psiT,pars = torchsolver.setup(dt,tN,zN,psiInitial,torchsolver.havercampSetpars)

            psiList = torchsolver.fullModelRun(dt,dts,dz,n,nt,psi,psiB,psiT,pars, torchsolver.havercampCfun,torchsolver.havercampKfun,torchsolver.havercampthetafun,flag,torchsolver.zeroFun)
            output = torchsolver.outputWrapper(psiList,flag,psiB,psiT)
            output = [tensor.detach().numpy() for tensor in output]
            
            # Prepare the data for saving
            data = {
                't': t,
                'output': output,
            }
            
            # Convert data to DataFrame to save as CSV
            df = pd.DataFrame([data])
            
            # Append the data to the CSV file
            df.to_csv(f, header=False, index=False)
            print(_)
    
    print("Data generation completed and saved to 'high_fidelity_training_data.csv'")

