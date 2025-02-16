import torch
from linalg import *

class LatOnePt(): # fuzzy zero
    r""" 
    One point function observable: O = z_i z^\dagger_j
    """
    def __init__(self,p,i,j,n,L):
        # lattice sides
        self.p = p
        # components
        self.i = i
        self.j = j
        # enc mask
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
        # enc mask
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
