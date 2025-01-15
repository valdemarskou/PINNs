


import torch_mpfd_solver as solver
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, tN, phi_initial):
        # Combine inputs and reshape for convolution
        x = torch.cat((tN, phi_initial), dim=1).unsqueeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)  # Flatten the output of the conv layers
        x = F.relu(self.fc1(x))
        delta_phi = self.fc2(x)
        return delta_phi


# Hybrid model run function incorporating the neural network
class HybridModel(nn.Module):

    #z,t,dz,n,nt,zN,psi,psiB,psiT,pars = solver.setup(dt,tN,zN,psiInitial)
    #output, err = solver.ModelRun(dt,dz,n,nt,psi,psiB,psiT,pars)

    def __init__(self, model_func, nn):
        super(HybridModel, self).__init__()
        self.model_func = model_func
        self.nn = nn

    def forward(self,dt,tN,zN,psi_initial):
        z,t,dz,n,nt,zN,psi,psiB,psiT,pars = solver.setup(dt,tN,zN,psi_initial)
        psi_approx_initial = self.model_func(tN, psi_initial)
        delta_psi = self.nn(tN, psi_initial)
        psi_approx = psi_approx_initial + delta_psi
        return psi_approx







#Downsampling training data
def downsamplefun(vec_fine,scaling_ratio):
    # Requires that fine vector comes from 
    #ratio = int((len(vec_fine)-1)/(len(vec_coarse)-1))
    #print(ratio)
    return vec_fine[::scaling_ratio]

class CustomDataset(Dataset):
    def __init__(self, data, downsamplefun, scaling_ratio):
        """
        Args:
            data (list of tuples): List of (tN, psiInitial, psiFinal) tuples.
            downsamplefun (function): Function to downsample psiInitial and psiFinal.
            scaling_ratio (float): Ratio used for downsampling.
        """
        self.data = [
            (tN, downsamplefun(psiInitial, scaling_ratio), downsamplefun(psiFinal, scaling_ratio))
            for tN, psiInitial, psiFinal in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tN, psiInitial, psiFinal = self.data[idx]
        return tN, psiInitial, psiFinal

