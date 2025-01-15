import torch 
from torch import nn
import torch.optim as optim

import numpy as np
from linalg import *
from utils import grab

import tqdm


class LatticeActionFunctional():
    def __init__(self,n):
        self.n = n
        self.identity = torch.eye(2*n + 2)
        sigma_y = torch.tensor([[0,-1j],[1j,0]]) # VS overall - sign! To discuss
        self.T = (
            self.identity + torch.block_diag(*[sigma_y for _ in range(n+1)])
                    ).cdouble()

    def S(self,phi,beta):
        """
        Parameters:
        -----------
        phi: torch.Tensor
            field evaluated on 2d square lattice
        beta: float
            coupling constant

        Returns:
        --------
        S: torch.Tensor
            action functional evaluated on lattice
        """
        assert phi.shape[-2] == 2*self.n + 2, f"phi has wrong (vector, real) dimension; expected {2*self.n + 2} but got {phi.shape[-2]}"

        L = torch.zeros_like(phi)

        for mu in [-3,-4]:
            z = ( phi.transpose(-1,-2) @ (self.T @ torch.roll(phi,-1,mu) ) ).squeeze(-1,-2)
            L +=  z * torch.conj_physical(z)
        
        return - beta * L.sum(dim=(-1,-2))


class ToyActionFunctional():
    def __init__(self,n):
        self.identity = torch.eye(2*n + 2)
        sigma_y = torch.tensor([[0,-1j],[1j,0]])
        self.T = (
            self.identity + torch.block_diag(*[sigma_y for _ in range(n+1)])
                    ).cdouble()

    def S(self,phi,beta):
        """
        phi = X, Y (samples, fields, dim, 1)
        """

        X = phi[:,0].cdouble()
        Y = phi[:,1].cdouble()

        hXY = ( X.transpose(-1,-2) @ (self.T @ Y) ).flatten()
        hYX =  ( Y.transpose(-1,-2) @ (self.T @ X) ).flatten()

        return - beta * hXY * hYX
    
class CP(nn.Module):

    def __init__(self,dim_C,action,deformation,observable,beta):
        """
    
        obs = observable
        a,b = (contour) deformation parameters 
        """
        super().__init__()

        # HYPERPARAMETERS
        self.beta = beta
        self.obs = observable
        self.deformation = deformation

        # ACTION
        self.S = action

    def deformed_obs(self,phi):
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

        tildeZ, detJ = self.deformation.complexify(phi) 

        Stilde = self.S(tildeZ)

        O = (self.obs(tildeZ) * detJ * torch.exp( - ( Stilde - S0 ) ) ) 

        return O

    def forward(self,phi):
        return self.deformed_obs(phi)
        
    def get_deformation_param(self):
        return self.deformation.get_param()    

def train(model,phi,epochs,loss_fct,lr=1e-4,split=0.7,batch_size=32):
    """
    Training.

    Parameters:
    -----------
    model: ZeroDModel
        model to train

    phi: torch.Tensor
        MCMC samples 

    epochs: int
        max number of epochs

    loss_fct: Callable
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

    a0: torch.Tensor
        deformation parameter before training

    af: torch.Tensor
        deformation parameter after training
    """
    # TRAIN-TEST SPLIT
    split_ = int(split*phi.shape[0])
    
    phi_train = phi[:split_]
    phi_val = phi[split_:]

    # OPTIMIZER
    optimizer_ = optim.Adam(model.parameters(),lr=lr)
    
    # MINI-BATCHING
    batch_size_ = batch_size

    observable = []
    observable_var = []
    losses_train = []
    losses_val = []
    anorm = []

    # a0 FOR COMPARISON
    a0 = model.get_deformation_param().copy() 

    # TRAINING
    for epoch in tqdm.tqdm(range(epochs)):
        optimizer_.zero_grad()

        # MINI-BATCHING
        minibatch = np.random.randint(low=0,high=len(phi_train),size=batch_size_)

        # TRAIN
        deformed_obs = model(phi_train[minibatch]) # .cdouble()

        loss_train = loss_fct(deformed_obs)
        loss_train.backward()

        with torch.no_grad():
            observable.append(grab(deformed_obs).mean())

            observable_var.append(grab(deformed_obs).var())

        optimizer_.step()

        # VALIDATION
        with torch.no_grad():
            deformed_obs_val = model(phi_val) # .cdouble()

            loss_val = loss_fct(deformed_obs_val)
            
        losses_train.append(grab(loss_train))
        losses_val.append(grab(loss_val))

        anorm.append(np.linalg.norm(model.get_deformation_param().ravel()))

    # GET DEFORMATION PARAMETER AFTER TRAINING 
    af = model.get_deformation_param() 

    return observable, observable_var, losses_train, losses_val, anorm, a0, af
