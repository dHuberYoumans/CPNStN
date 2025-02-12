import torch
from linalg import *

class ToyObs:
    def __init__(self):
        pass

    @staticmethod
    def fuzzy_one(phi):
        return torch.ones(phi.shape[0],dtype=torch.cdouble)
    
    @staticmethod
    def one_pt(phi,i,j,particle=0): # fuzzy zero
        """
        Observable z_i \\bar z_j

        Parameters:
        -----------
        phi: torch.tensor
            Batch of real fields (vectors (...,2n+1,1))

        i: int
            Component z_i

        j: int
            Component \\bar z_j

        particle: int, default 0
            Particle 0: z, 1: w
        """

        z, zbar = real2cmplx(phi[:,particle])

        O = (z[...,i,:]*zbar[...,j,:]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O
    
    @staticmethod
    def two_pt(phi,i,j):
        """
        Observable z_i zbar_j w_j wbar_i

        Parameters:
        -----------
        phi: torch.tensor
            Batch of real fields (...,2n+2,1)

        i: int
            Component z_i
        """

        z, zbar = real2cmplx(phi[:,0]) 
        w, wbar = real2cmplx(phi[:,1])

        O = (z[...,i,:]*zbar[...,j,:]*w[...,j,:]*wbar[...,i,:]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O

    def two_pt_full(phi):
       """
       Full two point function < |z^dagger w|^2 >

       Parameters:
       -----------
       phi: torch.tensor
            Batch of real fields (...,2n+2,1)
       """
       _, zbar = real2cmplx(phi[:,0]) 
       w, _ = real2cmplx(phi[:,1])

       O = torch.abs(inner(zbar,w))**2

       return O

class LatObs:
    def __init__(self):
        pass

    @staticmethod
    def fuzzy_one(phi):
        return torch.ones(phi.shape[0],dtype=torch.cdouble)

class LatOnePt(): # fuzzy zero
    r""" 
    One point function < z_i z^\dagger_j >
    """
    def __init__(self,p,i,j,n,L):
        self.p = p
        self.i = i
        self.j = j
        self.mask = torch.zeros(2*(n+1), L, L)
        self.mask[i,*p] += 1
        self.mask[j+(n + 1), *p] += 1

    def __call__(self, phi):
        x,y = self.p

        z, zbar = real2cmplx(phi[:,x,y])

        O = (z[:,self.i]*zbar[:,self.j]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O

class LatTwoPt(): # fuzzy zero
    r""" 
    Two point function < z_i(p) z^\dagger_j(p) w^\dagger_k(q) w_ell(q) >
    """
    def __init__(self,p,q,i,j,k,ell,n,L):
        # lattice sides
        self.p = p
        self.q = q
        # components
        self.i = i
        self.j = j
        self.k = k
        self.ell = ell
        self.mask = torch.zeros(2*(n+1), L, L)
        self.mask[i, *p] += 1
        self.mask[ell, *q] += 1
        self.mask[j+(n + 1), *p] += 1
        self.mask[k + (n+1), *q] += 1

    def __call__(self, phi):
        x1,y1 = self.p
        x2,y2 = self.q

        z, zbar = real2cmplx(phi[:,x1,y1])
        w, wbar = real2cmplx(phi[:,x2,y2])

        O = (z[:,self.i]*zbar[:,self.j]*wbar[:,self.k]*w[:,self.ell]).squeeze(-1) # (samples,)

        assert len(O.shape) == 1

        return O
