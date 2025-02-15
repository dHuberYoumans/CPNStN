import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm


def R(t):
    """
    batch of 2x2 rotation matrix

    Parameters:
    -----------
    t: Tensor 
        Rotation angles

    Returns:
    --------

    rot: Tensor
        Batch of rotation matrices around angle t
    """
    batch = t.shape
    rot = torch.stack(
        [torch.cos(t), -torch.sin(t),
        torch.sin(t), torch.cos(t)],
        dim=-1).reshape(*batch, 2, 2)

    return rot.double()

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
        vnew = R(theta).unsqueeze(0) @ v # R returns (batch,2,2)

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

    for a,b in pairs:
        idxs = torch.tensor([a,b])
        phi_old = phi[-1] # (2,dim,1)
        
        # PROPOSE NEW STATE
        theta = torch.randn(2)
        v = phi_old[:,idxs]
        vnew = R(theta) @ v
        phi_new = phi_old.detach().clone()
        phi_new[:,idxs] = vnew # (2,dim,1)
        phi_new /= torch.linalg.norm(phi_new,dim=1).unsqueeze(-1) # normalize
        norm = torch.linalg.norm(phi_new,dim=1)
        assert torch.allclose(norm,torch.tensor(1.).double(),atol=1e-12), "new phi not normalized"

        # ACCEPTENCE PROBABILITIES
        A = torch.minimum(torch.tensor(1.0), f(phi_new) / (f(phi_old)) )

        # ACCEPTED / REJECT
        p = torch.rand(1) 
        
        if p <= A : 
            phi.append(phi_new)
            alpha += 1
        else:
            phi.append(phi_old)

    return alpha / len(pairs)

def create_samples_II(n, phi0, beta, N_steps,burnin,k):
    """
    Monte Carlo Markov Chain (MCMC) sampling of the toy model using parallel updates

    Parameters:
    -----------
    n: int
        complex dimension CP^n 

    phi0: Tensor
        Initial field configuration. Shape (2,dimR,1)

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
    identity = torch.eye(2*(n + 1))
    sigma_y = torch.tensor([[0,1j],[-1j,0]])
    T = identity + torch.block_diag(*[sigma_y for _ in range(n+1)]) # dtype = cfloat 
    
    # DEF ACTION
    def h(Z,W):
        return ( Z.transpose(-1,-2).cfloat() @ ( T @ W.cfloat()) ).flatten() # (samples,)

    def S(phi,beta):
        return - beta * torch.abs(h(phi[0],phi[1]))**2
    
    phi = [phi0]

    alpha = 0 # acceptance rate

    for _ in tqdm(range(N_steps)): # tqdm for progress bar
        alpha += metropolis(phi=phi,pairs=pairs,f = lambda phi: torch.exp(- S(phi,beta)) )
    
    acception_rate = alpha / (N_steps) #(2*N_steps) 
    return (torch.stack(phi,dim=0)[burnin::k], acception_rate)

def create_samples_seq(n, phi0, beta, N_steps,burnin,k):
    """
    Monte Carlo Markov Chain (MCMC) sampling of the toy model using sequential update

    Parameters:
    -----------
    n: int
        complex dimension CP^n 

    phi0: Tensor 
        Initial field configuration. Shape (2,dimR,1)

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
    
    phi = phi0 

    Z = [phi0[0]]
    W = [phi0[1]]

    alpha = 0 # acceptance rate

    for _ in tqdm(range(N_steps)): # tqdm for progress bar
        alpha += sweep_sphere(phi=Z,pairs=pairs,f=lambda z: torch.exp( -S(z,W[-1],beta))) # w fixed
        alpha += sweep_sphere(phi=W,pairs=pairs,f=lambda w: torch.exp( -S(Z[-1],w,beta))) # z fixed
    
    samples_Z, samples_W = (torch.stack(Z,dim=0)[burnin::k], torch.stack(W,dim=0)[burnin::k])

    acception_rate = alpha / (2* N_steps)

    return  (torch.stack([samples_Z, samples_W],dim=1), acception_rate)

def generate_toy_samples(n,beta,N_steps = 10_000, burnin = 1_000, skip = 10, mode = 'II'):
    """ GENERATING SAMPLES """

    phi0 = torch.randn(2,(2*n + 2),1).double()
    phi0 /= torch.linalg.vector_norm(phi0, dim=1, keepdim=True)

    print(f"creating samples ({mode = })... \n")

    fn_map = {
            'II': create_samples_II,
            'seq': create_samples_seq
    }

    phi, alpha = fn_map[mode](n=n,phi0=phi0,beta=beta,N_steps=N_steps,burnin=burnin,k=skip)

    print("\ndone")
    print(f"\n{phi.shape = }\t{alpha = }\n")
    print("\nsaving ...")

    torch.save(phi, f"samples_n{n}_b{beta}_m{mode}.dat")

    print("\ndone")


