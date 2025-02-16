import torch
from torch import nn
from linalg import *
from utils import *

class NNHom(nn.Module):
    def __init__(self,neural_net,n,safe_grab=False):
        super().__init__()

        self.nn = neural_net
        
        self.n = n # CP(n)
        self.dim_g = n**2 + 2*n

        # self.a = self.nn().permute(1,2,0) # (Lx,Ly,dim_g)

        self.identity = torch.eye(2*n + 2)

        self.su_n = LieSU(n+1)

        self.safe_grab = safe_grab

    # DEFORM 
    def complexify(self, phi, mask):
        assert phi.shape[-2] == 2*self.n + 2, f"phi has wrong (vector, real) dim. Expected {2*self.n + 2}, got {phi.shape[-2]}"

        X = phi # (samples, Lx,Ly,dim_R,1)
        dtype = X.dtype 

        self.a = self.nn(mask).permute(1,2,0)

        a_ = rho(1j*self.su_n.embed(self.a)).to(dtype) # assuming Hermitian su(n) generators 

        outer_XX = X @ X.transpose(-1,-2)

        # DEFORMATION / COMPLEXIFICATION
        Y = a_ @ X 
        inner_YY = Y.transpose(-1,-2) @ Y

        lam = torch.sqrt(1 + inner_YY)

        tildeZ = X * lam + 1j*Y 

        # JACOBIAN
        J = self.identity*lam - outer_XX @ (a_ @ a_) / lam + 1j*a_
        det = torch.det(J) # (samples, particles) -> multiply 

        detJ = det / (lam.squeeze(dim=(-1,-2))**2) # incl. extra factor from delta fct

        Lx = X.shape[-4]
        Ly = X.shape[-3]

        detJ = torch.prod(detJ.view(-1,Lx*Ly),dim=-1) # total Jacobian (samples,)

        assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 
    
        return tildeZ, detJ
    
    def get_param(self):
        return grab(self.a,safe=self.safe_grab)
    
