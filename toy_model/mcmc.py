import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm


def R(t):
    """
    2x2 roation matrix 

    :param t: rotation angle
    :type t: float

    :returns: rotation matrix np.array([[cos(t),-sin(t)],[sin(t),cos(t)]])
    :rtype: np.ndarray
    """

    c = torch.cos(t)
    s = torch.sin(t)

    return torch.tensor([[c,-s],[s,c]]).double()

def sweep_sphere(phi,pairs,f):  
    """
    sweeping the sphere

    :param n: complex dimension CP^n
    :type n: int
    """

    alpha = 0 # accepted

    # LOOP OVER COMPONENTS 
    for pair in pairs:
        a,b = pair 
        phi_old = phi[-1] 

        # SAMPLE ROTATION ANGLE
        theta = torch.randn(1)

        # ROTATE COMPONENT
        v = torch.tensor([phi_old[a],phi_old[b]])
        vnew = R(theta) @ v

        phi_new = phi_old.detach().clone()
        phi_new[a] = vnew[0]
        phi_new[b] = vnew[1]
        phi_new /= torch.linalg.vector_norm(phi_new, keepdim=True) # normalize, else only 1e-8 precision -> adds up

        norm_ = (phi_new.transpose(-1,-2) @ phi_new)[0]
        assert torch.allclose(norm_,torch.tensor(1.,dtype=torch.double),atol=1e-12), f"phi not normalized for pair {pair}: {norm_.item()}"
    
        # ACCEPTENCE PROBABILITIES
        A = torch.minimum(torch.tensor(1.0), f(phi_new.unsqueeze(0)) / (f(phi_old.unsqueeze(0))) ) # unsqueeze -> (smaples,2n+2,1)

        # ACCEPTED / REJECT
        p = torch.rand(1) 
        
        if p <= A : 
            phi.append(phi_new)
            alpha += 1
        else:
            phi.append(phi_old)

    return alpha / len(pairs)

def metropolis(phi,pairs,f) -> float:
    """
    Metropolis Hasitings step of MCMC 

    Parameters:
    -----------
    n: int
        complex dimension CP^n 

    phi0 list[Tensor] 
        List of initial field configuration. Shape of Tensor (2,dimR,1)

    f: Callable
        Probability distribution one samples from
    """
    alpha = 0

    for pair in pairs:
        a,b = pair
        phi_old = phi[-1] # (2,dim,1)
        Z_old = phi_old[0]
        W_old = phi_old[1]

        # ROTATE Z
        theta = torch.randn(1)
        v = torch.tensor([Z_old[a],Z_old[b]])
        vnew = R(theta) @ v
        
        # ROTATE W
        theta = torch.randn(1)
        u = torch.tensor([W_old[a],W_old[b]])
        unew = R(theta) @ u

        Z_new = Z_old.detach().clone()
        Z_new[a] = vnew[0]
        Z_new[b] = vnew[1]
        Z_new /= torch.linalg.vector_norm(Z_new, keepdim=True) # normalize, else only 1e-8 precision -> adds up

        W_new = W_old.detach().clone()
        W_new[a] = unew[0]
        W_new[b] = unew[1]
        W_new /= torch.linalg.vector_norm(W_new, keepdim=True) 


        Z_norm = (Z_new.transpose(-1,-2) @ Z_new)[0]
        W_norm = (W_new.transpose(-1,-2) @ W_new)[0]
        assert torch.allclose(Z_norm,torch.tensor(1.,dtype=torch.double),atol=1e-12), f"phi not normalized for pair {pair}: {Z_norm.item()}"
        assert torch.allclose(W_norm,torch.tensor(1.,dtype=torch.double),atol=1e-12), f"phi not normalized for pair {pair}: {W_norm.item()}"
   
        phi_new = torch.stack([Z_new,W_new],dim=0) # (2,dim,1)

        # ACCEPTENCE PROBABILITIES
        A = torch.minimum(torch.tensor(1.0), f(Z_new.unsqueeze(0),W_new.unsqueeze(0)) / (f(Z_old.unsqueeze(0),W_old.unsqueeze(0))) ) # unsqueeze -> (smaples,2n+2,1)

        # ACCEPTED / REJECT
        p = torch.rand(1) 
        
        if p <= A : 
            phi.append(phi_new)
            alpha += 1
        else:
            phi.append(phi_old)

    return alpha / len(pairs)


def create_samples(n, phi0, beta, N_steps,burnin,k):
    """
    Monte Carlo Markov Chain (MCMC) sampling of the toy model

    Parameters:
    -----------
    n: int
        complex dimension CP^n 

    phi0: list[Tensor] 
        List of initial field configuration. Shape of Tensor (2,dimR,1)

    beta: float
        Coupling constant

    N_steps: int
        Number Monte Carlo steps

    burnin: int 
        Burnin

    k: int
        Return only every k-th element (to reduce correlation)
    """
    
    N = 2*n + 2 
    pairs = list(combinations(np.arange(N),2)) # indices of pairs to rotate

    # LIN ALG NEEDED FOR ACTION
    id = torch.eye(2*(n + 1))
    sigma_y = torch.tensor([[0,1j],[-1j,0]])
    T = id + torch.block_diag(*[sigma_y for _ in range(n+1)]) # dtype = cfloat 
    
    # DEF ACTION
    def h(Z,W):
        return ( Z.transpose(-1,-2).cfloat() @ ( T @ W.cfloat()) ).flatten() # (samples,)

    def S(Z,W,beta):
        return - beta * torch.abs(h(Z,W))**2
    
    # Z0 = phi0[0] # (samples, dimR, 1)
    # W0 = phi0[1]

    # Z = [Z0] 
    # W = [W0]

    phi = phi0

    alpha = 0 # acceptance rate

    for _ in tqdm(range(N_steps)): # tqdm for progress bar
        # alpha += sweep_sphere(phi=Z,pairs=pairs,f=lambda z: torch.exp( -S(z,W[-1],beta))) # w fixed
        # alpha += sweep_sphere(phi=W,pairs=pairs,f=lambda w: torch.exp( -S(Z[-1],w,beta))) # z fixed
        alpha += metropolis(phi=phi,pairs=pairs,f = lambda Z,W: torch.exp(- S(Z,W,beta)) )
    
    # samples_Z, samples_W = (torch.stack(Z,dim=0)[burnin::k], torch.stack(W,dim=0)[burnin::k])

    acception_rate = alpha / (N_steps) #(2*N_steps) 
    return (torch.stack(phi,dim=0), acception_rate)

    # return  (torch.stack([samples_Z, samples_W],dim=1), acception_rate)
