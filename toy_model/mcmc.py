import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm


def R(t:float) -> torch.tensor:
    """
    2x2 roation matrix 

    :param t: rotation angle
    :type t: float

    :returns: rotation matrix np.array([[cos(t),-sin(t)],[sin(t),cos(t)]])
    :rtype: np.ndarray
    """

    c = torch.cos(t)
    s = torch.sin(t)

    return torch.tensor([[c,-s],[s,c]])

def sweep_sphere(phi:list,pairs:list,f) -> float:  
    """
    sweeping the sphere

    :param n: complex dimension CP^n
    :type n: int
    """

    alpha = 0 # number accepted

    # LOOP OVER COMPONENTS 
    for pair in pairs:
        a,b = pair # unpack indices
        phi_old = phi[-1] # current state 

        # SAMPLE 2x2 ROTATION ANGLE
        theta = torch.randn(1)

        # ROTATE COMPONENT
        v = torch.tensor([phi_old[a],phi_old[b]])
        vnew = R(theta) @ v

        # PROPOSE NEW STATE
        phi_new = phi_old.detach().clone()

        phi_new[a] = vnew[0]
        phi_new[b] = vnew[1]
    
        # ACCEPTENCE PROBABILITIES
        A = torch.minimum(torch.tensor(1), f(phi_new.unsqueeze(0)) / (f(phi_old.unsqueeze(0))) ) # unsqueeze since f takes list of vectors: shape = (smaples,2n+2,1)

        # CHECK IF ACCEPTED
        p = torch.rand(1) # draw vector of uniform rnds
        
        if p <= A : # accept if p < A
            phi.append(phi_new)
            alpha += 1
        else:
            phi.append(phi_old)

    return alpha / len(pairs)

def create_samples(n:int, phi0:torch.tensor, beta:float, N_steps:int,burnin:int,k:int) -> torch.tensor:
    """
    :param n: complex dimension CP^n 
    :type n: int

    :param phi0: initial state
    :type phi0:
    """
    
    N = 2*n + 2 # real dimension (before quotient by C^*)
    pairs = list(combinations(np.arange(N),2)) # indices of pairs to rotate

    # LIN ALG NEEDED FOR ACTION
    id = torch.eye(2*(n + 1))
    sigma_y = torch.tensor([[0,-1j],[1j,0]])
    T = id + torch.block_diag(*[sigma_y for _ in range(n+1)]) # dtype = cfloat 
    
    # DEF ACTION
    def h(Z,W):
        return ( Z.transpose(-2,-1).cfloat() @ ( T @ W.cfloat()) ).flatten()

    def S(Z,W,beta):
        return (-beta * h(Z,W) * h(W,Z)).real
    
    # SETUP
    Z0 = phi0[0] # (nb fields, dim, 1)
    W0 = phi0[1]

    Z = [Z0] 
    W = [W0]

    alpha = 0 # number accepted

    expSw = lambda z: torch.exp( -S(z,W[-1],beta))    # w fixed
    expSz = lambda w: torch.exp( -S(Z[-1],w,beta))    # z fixed

    for _ in tqdm(range(N_steps)): # tqdm for progress bar

        alpha += sweep_sphere(phi=Z,pairs=pairs,f=expSw) 
        alpha += sweep_sphere(phi=W,pairs=pairs,f=expSz)
    
    samples_Z, samples_W = (torch.stack(Z,dim=0)[burnin::k], torch.stack(W,dim=0)[burnin::k])

    acception_rate = alpha / (2*N_steps) 

    return  (torch.stack([samples_Z, samples_W],dim=1), acception_rate)
