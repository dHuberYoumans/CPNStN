import torch
from linalg import *

class ToyOnePt:
    """
    One-point function observable: O = z_i z*_j 
    
    Parameters:
    -----------
    i: int
        vector component z_i
    j: int 
        vector component z*_j
    particle: int
        which particle (0 => z, 1 => w)
    """

    def __init__(self,i,j,particle):
        self.i = i
        self.j = j
        self.particle = particle

    def __call__(self, phi):
        z, zbar = real2cmplx(phi[:,self.particle])

        O = (z[...,self.i,:]*zbar[...,self.j,:]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O

class ToyTwoPt:
    """
    Two-point function observable: O = z_i z*_j w*_j w_i

    Parameters:
    -----------
    i: int
        vector component z_i / w_i
    j: int 
        vector component z*_j / w*_j
    """

    def __init__(self,i,j):
        self.i = i
        self.j = j

    def __call__(self,phi):
        
        z, zbar = real2cmplx(phi[:,0]) 
        w, wbar = real2cmplx(phi[:,1])

        O = (z[...,self.i,:]*zbar[...,self.j,:]*w[...,self.j,:]*wbar[...,self.i,:]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O

class ToyFullTwoPt:
    r""" 
    Full two point function observable: O = |z* w|^2 = \sum_{i,j} z*_i w_i w*_j z_j 

    """
    def __init__(self) -> None:
        pass

    def __call__(self,phi):
       _, zbar = real2cmplx(phi[:,0]) 
       w, _ = real2cmplx(phi[:,1])

       O = torch.abs(inner(zbar,w))**2

       return O

class ToyFuzzyOne:
    """Fuzzy one:  O = 1"""

    def __init__(self) -> None:
        pass

    def __call__(self,phi):
        return torch.ones(phi.shape[0],dtype=torch.cdouble)

# class ToyObs:
    # def __init__(self):
        # pass
# 
    # @staticmethod
    # def fuzzy_one(phi):
        # return torch.ones(phi.shape[0],dtype=torch.cdouble)
    # 
    # @staticmethod
    # def one_pt(phi,i,j,particle=0): # fuzzy zero
        # """
        # Observable z_i \\bar z_j
# 
        # Parameters:
        # -----------
        # phi: torch.tensor
            # Batch of real fields (vectors (...,2n+1,1))
# 
        # i: int
            # Component z_i
# 
        # j: int
            # Component \\bar z_j
# 
        # particle: int, default 0
            # Particle 0: z, 1: w
        # """
# 
        # z, zbar = real2cmplx(phi[:,particle])
# 
        # O = (z[...,i,:]*zbar[...,j,:]).squeeze(-1) # (samples,)
# 
        # assert len(O.shape) == 1
# 
        # return O
    # 
    # @staticmethod
    # def two_pt(phi,i,j):
        # """
        # Observable z_i zbar_j w_j wbar_i
# 
        # Parameters:
        # -----------
        # phi: torch.tensor
            # Batch of real fields (...,2n+2,1)
# 
        # i: int
            # Component z_i
        # """
# 
        # z, zbar = real2cmplx(phi[:,0]) 
        # w, wbar = real2cmplx(phi[:,1])
# 
        # O = (z[...,i,:]*zbar[...,j,:]*w[...,j,:]*wbar[...,i,:]).squeeze(-1) # (samples,)
# 
        # assert len(O.shape) == 1
# 
        # return O
# 
    # def two_pt_full(self,phi):
       # """
       # Full two point function < |z^dagger w|^2 >
# 
       # Parameters:
       # -----------
       # phi: torch.tensor
            # Batch of real fields (...,2n+2,1)
       # """
       # _, zbar = real2cmplx(phi[:,0]) 
       # w, _ = real2cmplx(phi[:,1])
# 
       # O = torch.abs(inner(zbar,w))**2
# 
       # return O
# 
