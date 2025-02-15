import torch 
from torch import nn
import torch.optim as optim

import numpy as np
from linalg import *
from deformations import *
from utils import *

import tqdm

class ToyActionFunctional():
    def __init__(self,n):
        self.identity = torch.eye(2*n + 2)
        sigma_y = torch.tensor([[0,1j],[-1j,0]])
        self.T = ( self.identity + torch.block_diag(*[sigma_y for _ in range(n+1)]) ).cdouble()

    def action(self,phi,beta):
        """
        phi = X, Y (samples, fields, dim, 1)
        """

        X = phi[:,0].cdouble()
        Y = phi[:,1].cdouble()

        hXY = inner(X, (self.T @ Y))
        hYX = inner(Y, (self.T @ X))

        return - beta * hXY * hYX
    
class CP(nn.Module):
    def __init__(self,n,action,deformation,beta):
        """
    
        obs = observable
        a,b = (contour) deformation parameters 
        """
        super().__init__()

        # HYPERPARAMETERS
        self.beta = beta
        self.deformation = deformation

        # ACTION
        self.S = action

    def Otilde(self, obs, phi):
        """
        Computes the operator corresponding to the deformed observable inside the undeformed path integral: Jac * O(tilde phi) exp[ - ( S( tilde phi) - S(phi) ) ]

        Parameters:
        -----------

        phi: torch.Tensor
            MCMC samples

        a: torch.Tensor
            deformation parameter

        Returns:
        --------
        O: torch.Tensor
            Operator corresponding to the deformed observable inside the undeformed path integral 
            
        """

        S0 = self.S(phi)

        phi_tilde, detJ = self.deformation.complexify(phi) 
 
        Stilde = self.S(phi_tilde)

        O = (obs(phi_tilde) * detJ * torch.exp( - ( Stilde - S0 ) ) ) 

        return O
    
    def forward(self, obs, phi):
        return self.Otilde(obs, phi)
        
    def get_deformation_param(self):
        return self.deformation.get_param()    

def train(ddp_model, model, obs, phi, epochs, loss_fn, lr=1e-4, split=0.7, batch_size=32):
    """
    Training.

    Parameters:
    -----------
    model: ZeroDModel
        model to train

    observable: Observable
        observable

    phi: torch.Tensor
        MCMC samples 

    epochs: int
        max number of epochs

    loss_fn: Callable
        loss function 

    lr: float, optional, default = 1e-4
        learning rate

    optimizer: torch.optim.Optimizer, optional, default = None 
        optimizer, if None then set to torch.optim.Adam

    split: flaot, optional, default = 0.7
        percentage of splitting into training and validation set

    Returns:
    --------

    observable: torch.Tensor
        expectation value of the observable

    observable_var: torch.Tensor
        variance of the observable

    losses_train: list[float]
        losses of training set

    losses_val: list[float]
        losses of validation set

    anorm: list[float]
        vector/matrix norm of deformation parameter

    a: torch.Tensor
        deformation parameter after training
    """
    
    # TRAIN-TEST SPLIT
    split_ = int(split*phi.shape[0])
    
    phi_train = phi[:split_]
    phi_val = phi[split_:]

    # OPTIMIZER
    optimizer_ = optim.Adam(ddp_model.parameters(), lr=lr)
    
    # MINI-BATCHING
    batch_size_ = batch_size

    observable = []
    observable_var = []
    losses_train = []
    losses_val = []
    anorm = []

    # TRAINING
    for i in tqdm.tqdm(range(epochs)):
        optimizer_.zero_grad()

        # MINI-BATCHING
        minibatch = np.random.randint(low=0, high=len(phi_train), size=batch_size_)
        phi_batched = phi_train[minibatch]

        # TRAIN
        Otilde = ddp_model(obs, phi_batched) 
        loss_train = loss_fn(Otilde)
        losses_train.append((i,grab(loss_train)))
        loss_train.backward()

        with torch.no_grad():
            observable.append((i,grab(Otilde).mean()))
            observable_var.append((i,grab(Otilde).var()))

        optimizer_.step()

        # VALIDATION
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                Otilde_val = model(obs, phi_val) 

                loss_val = loss_fn(Otilde_val)
                losses_val.append((i+1,grab(loss_val)))
                       

        anorm.append(np.linalg.norm(model.get_deformation_param().ravel()))

    # GET DEFORMATION PARAMETER AFTER TRAINING 
    a = model.get_deformation_param() 

    return observable, observable_var, losses_train, losses_val, anorm, a
