import torch 
from torch import nn
import torch.optim as optim


import numpy as np
from linalg import *
from utils import *

import tqdm


class LatticeActionFunctional():
    def __init__(self,n,beta):
        self.n = n
        self.beta = beta
        self.identity = torch.eye(2*n + 2)
        sigma_y = torch.tensor([[0,1j],[-1j,0]]) 
        self.T = (
            self.identity + torch.block_diag(*[sigma_y for _ in range(n+1)])
                    ).cdouble()

    def action(self,phi):
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

        phi = phi.cdouble()
        S = torch.zeros(phi.shape[0],dtype=torch.cdouble)

        for mu in [-3,-4]:
            phi_fwd = torch.roll(phi,-1,dims=mu)
            
            h = inner(phi,self.T @ phi_fwd)
            hbar = inner(phi_fwd,self.T @ phi)

            S += (h*hbar).sum(dim=(-1,-2))
            
        return - self.beta * S
    
    def action_centered(self,phi):
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
        
        S = torch.zeros(phi.shape[0],dtype=torch.cdouble)

        for mu in [-3,-4]:
            phi_fwd = torch.roll(phi,-1,dims=mu)

            h = inner(phi, self.T @ phi_fwd)
            hbar = inner(phi_fwd, self.T @ phi)

            S += (h*hbar - 1.0).sum(dim=(-1,-2))
            
        return - self.beta * S
    
    def u(self,phi):
        assert phi.shape[-2] == 2*self.n + 2, f"phi has wrong (vector, real) dimension; expected {2*self.n + 2} but got {phi.shape[-2]}"

        L = phi.shape[-3]
        V = L**2

        u_ = torch.zeros(phi.shape[:-2],dtype=torch.cdouble)

        for mu in [-3,-4]:
            z_fwd = ( phi.transpose(-1,-2) @ (self.T @ torch.roll(phi,-1,dims=mu) ) ).squeeze(-1,-2) 
            zbar_fwd = ( torch.roll(phi,-1,dims=mu).transpose(-1,-2) @ (self.T @ phi ) ).squeeze(-1,-2)  
            u_ +=  1.0 - z_fwd * zbar_fwd

        return u_.sum(dim=(-1,-2)) / V

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

    def Otilde(self,phi):
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

        O = (self.obs(phi_tilde) * detJ * torch.exp( - ( Stilde - S0 ) ) ) 

        return O
    
    def forward(self,phi):
        return self.Otilde(phi)
        
    def get_deformation_param(self):
        return self.deformation.get_param()    

def train(ddp_model,model,phi,epochs,loss_fct,lr=1e-4,split=0.7,batch_size=32):
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
    optimizer_ = optim.Adam(ddp_model.parameters(),lr=lr)
    
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
    for i in tqdm.tqdm(range(epochs)):
        optimizer_.zero_grad()

        # MINI-BATCHING
        minibatch = np.random.randint(low=0,high=len(phi_train),size=batch_size_)
        phi_batched = phi_train[minibatch]
        rotate_lattice(phi_batched)

        # TRAIN
        Otilde = ddp_model(phi_batched) 
        loss_train = loss_fct(Otilde)
        losses_train.append((i,grab(loss_train)))
        loss_train.backward()

        with torch.no_grad():
            observable.append((i,grab(Otilde).mean()))
            observable_var.append((i,grab(Otilde).var()))

        optimizer_.step()

        # VALIDATION
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                Otilde_val = model(phi_val) 

                loss_val = loss_fct(Otilde_val)
                losses_val.append((i+1,grab(loss_val)))
                       

        anorm.append(np.linalg.norm(model.get_deformation_param().ravel()))

    # GET DEFORMATION PARAMETER AFTER TRAINING 
    af = model.get_deformation_param() 

    return observable, observable_var, losses_train, losses_val, anorm, a0, af

def rotate_lattice(phi_mini_batched):
    lattice_shape = phi_mini_batched.shape[1:-2]

    for i in range(len(phi_mini_batched)):
        dx = tuple(np.random.randint(L) for L in lattice_shape)
        phi_mini_batched[i] = torch.roll(phi_mini_batched[i],dx,dims=tuple(range(len(dx))))
