import torch
from torch import nn
from linalg import *
from utils import *

class Linear(nn.Module):
    def __init__(self,a0,n,mode="0D"):
        super().__init__()

        assert a0.shape[-2] == 2*n + 2, f"expected (vector, real) dim {2*n + 2}, got {a0.shape[-2]}"

        self.a = nn.Parameter(a0)
        self.identity = torch.eye(2*n + 2)

        self.mode = mode
        

    # DEFORM 
    def complexify(self,phi):
        X = phi

        inner_XX = X.transpose(-1,-2) @ X
        outer_XX = X @ X.transpose(-1,-2)
        inner_aX = self.a.transpose(-1,-2) @ X
        outer_aX = self.a @ X.transpose(-1,-2) 

        # DEFORMATION / COMPLEXIFICATION
        Y = self.a - inner_aX / inner_XX * X 

        inner_YY = Y.transpose(-1,-2) @ Y
        lam = torch.sqrt(1 + inner_YY) 

        tildeZ = X * lam + 1j*Y 

        # JACOBIAN
        M = - self.identity * inner_aX / inner_XX - outer_aX / inner_XX +  2*inner_aX * outer_XX / inner_XX**2

        J = self.identity*lam + (M @ Y) @ X.transpose(-1,-2) / lam + 1j*M

        det = torch.det(J) # (samples, particles) -> multiply
        detJ = det / lam.squeeze(dim=(-1,-2))**2 

        if self.mode == "0D":
            detJ = torch.prod(detJ,dim=-1) # total Jacobian (samples,)
        elif self.mode == "2D":
            Lx = detJ.shape[-2]
            Ly = detJ.shape[-1]
            detJ = torch.prod(detJ.view(-1,Lx*Ly),dim=-1) # total Jacobian (samples,)

        assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 

        return tildeZ, detJ
    
    def get_param(self):
        return grab(self.a)

class Homogeneous(nn.Module):
    def __init__(self,a0,n,mode="0D"):
        super().__init__()

        assert a0.shape[-1] == n**2 + 2*n

        self.a = nn.Parameter(a0)
        self.n = n # CP(n)
        self.identity = torch.eye(2*n + 2)

        self.su_n = LieSU(n+1)

        self.mode = mode

    # DEFORM 
    def complexify(self,phi):
        assert phi.shape[-2] == 2*self.n + 2, f"phi has wrong (vector, real) dim. Expected {2*self.n + 2}, got {phi.shape[-2]}"

        X = phi
        dtype = X.dtype # I don't like this!

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

        if self.mode == "0D":
            detJ = torch.prod(detJ,dim=-1) # total Jacobian (samples,)
        elif self.mode == "2D":
            Lx = detJ.shape[-2]
            Ly = detJ.shape[-1]
            detJ = torch.prod(detJ.view(-1,Lx*Ly),dim=-1) # total Jacobian (samples,)

        assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 
    
        return tildeZ, detJ
    
    def get_param(self):
        return grab(self.a)