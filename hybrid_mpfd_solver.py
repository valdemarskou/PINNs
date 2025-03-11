


import torch_mpfd_solver as torchsolver
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CorrectionNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        """
        in_channels: number of input channels 
            (e.g., 1 channel for the state + 1 channel for dt)
        out_channels: number of output channels for the correction
            (e.g., same as number of state channels you want to predict)
        """
        super(CorrectionNet, self).__init__()
        
        # Example: 2D convolution layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, height, width]
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        # The output is the correction with shape [batch_size, out_channels, height, width]
        return x

#Instantiate correction_net 
correction_net = CorrectionNet(in_channels=2, out_channels=1)


def hybridSolverOneStepModelRun(dt,dz,n,psi,psiB,psiT,pars,Cfun,Kfun,thetafun,sink,correction_net):
    """
    s: [B, 1, H, W] (for example)
    dt: [B] or [B, 1]
    correction_net: instance of CorrectionNet
    Returns: corrected next state
    
    """
    # 1. Baseline solver
    baseline_state = torchsolver.dirichletOneStepModelRun(dt,dz,n,psi,psiB,psiT,pars,Cfun,Kfun,thetafun,sink)
    
    # 2. Prepare input for correction net by concatenating dt
    # If dt is shape [B], expand to [B, 1, H, W] to match image shape
    B, C, H, W = psi.shape
    dt_reshaped = dt.view(B, 1, 1, 1).expand(B, 1, H, W)
    
    # Concatenate along the channel dimension
    cnn_input = torch.cat([psi, dt_reshaped], dim=1)  # shape [B, C+1, H, W]
    
    correction = correction_net(cnn_input)
    
    # 3. Sum baseline + correction
    next_state = baseline_state + correction
    return next_state
