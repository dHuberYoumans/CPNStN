import torch
from linalg import *
       
class Linear():
    def __init__(self):
        pass

    # DEFORM 
    def complexify(self,phi,a):
        X = phi
        dim_R = X.shape[-2]
        id = torch.eye(dim_R)

        inner_XX = X.transpose(-1,-2) @ X
        outer_XX = X @ X.transpose(-1,-2)
        inner_aX = a.transpose(-1,-2) @ X
        outer_aX = a @ X.transpose(-1,-2) 

        # DEFORMATION / COMPLEXIFICATION
        Y = a - inner_aX / inner_XX * X 

        inner_YY = Y.transpose(-1,-2) @ Y
        lam = torch.sqrt(1 + inner_YY) 

        tildeZ = X * lam + 1j*Y 

        # JACOBIAN
        M = - id * inner_aX / inner_XX - outer_aX / inner_XX +  2*inner_aX * outer_XX / inner_XX**2

        J = id*lam + (M @ Y) @ X.transpose(-1,-2) / lam + 1j*M

        det = torch.det(J) # (batch, #particles) -> multiply
        detJ = det / lam.squeeze(dim=(-1,-2))**2 

        detJ = torch.prod(detJ,dim=-1) # total Jacobian (batch,)

        assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 

        return tildeZ, detJ

class Homogeneous():
    def __init__(self):
        pass

    # DEFORM 
    def complexify(self,phi,a):
        X = phi
        dim_R = X.shape[-2]
        dim_C = dim_R // 2
        id = torch.eye(dim_R)

        if not hasattr(self, 'su_n'):
            self.su_n = LieSU(dim_C)

        a_ = rho(1j*self.su_n.embed(a)).cdouble() # assuming Hermitian su_n generators 

        outer_XX = X @ X.transpose(-1,-2)

        # DEFORMATION / COMPLEXIFICATION
        Y = a_ @ X 
        inner_YY = Y.transpose(-1,-2) @ Y

        lam = torch.sqrt(1 + inner_YY)

        tildeZ = X * lam + 1j*Y 

        # JACOBIAN
        J = id*lam - outer_XX @ (a_ @ a_) / lam + 1j*a_
        det = torch.det(J) # (samples, #particles) -> multiply 
        detJ = det / lam.squeeze(dim=(-1,-2))**2 

        detJ = torch.prod(detJ,dim=-1) # total Jacobian (samples,)

        assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 
    
        return tildeZ, detJ

# class Torus():
#     def __init__(self):
#         pass

#     # DEFORM 
#     def complexify(self,phi,a):
#         X = phi
#         dimR = X.shape[-2]
#         dimC = dimR // 2
#         id = torch.eye(dimR)

#         su_n = LieSU(dimC)
#         a_ = rho(su_n.embed(a)).cdouble()

#         outer_XX = X @ X.transpose(-1,-2)

#         # DEFORMATION / COMPLEXIFICATION
#         Y = a_ @ X 
#         inner_YY = Y.transpose(-1,-2) @ Y

#         lam = torch.sqrt(1 + inner_YY)

#         tildeZ = X * lam + 1j*Y 

#         # JACOBIAN
#         J = id*lam + (a_ @ a_) @ outer_XX / lam - 1j*a_
#         det = torch.det(J) # (samples, #particles) -> multiply
#         detJ = det / lam.squeeze(dim=(-1,-2))**2 

#         detJ = torch.prod(detJ,dim=-1) # total Jacobian (samples,)

#         assert len(detJ.shape) == 1, f'detJ has wrong dim: {detJ.shape} but must be 1' 
    
#         return tildeZ, detJ

