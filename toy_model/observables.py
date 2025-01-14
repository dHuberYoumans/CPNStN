import torch

# OBSERVABLES
def real2cmplx(phi):
    z = phi[...,::2,:] + 1j*phi[...,1::2,:]
    zbar = phi[...,::2,:] - 1j*phi[...,1::2,:]

    return (z,zbar)
    
def cmplx2real(z):
    re = z.real
    im = z.imag
    Xshape = [elem for elem in z.shape]
    Xshape[-2] *= 2

    X = torch.zeros(*Xshape)
    X[...,::2,:] = re
    X[...,1::2,:] = im

    return X

def fuzzy_one(phi):
    return torch.ones(phi.shape[0],dtype=torch.cdouble)

def fuzzy_zero(phi,i,j):
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
    """

    z, zbar = real2cmplx(phi[:,0])

    O = (z[...,i,:]*zbar[...,j,:]).squeeze(-1) # (samples,)

    assert len(O.shape) == 1

    return O

def two_pt(phi,i,j):
    """
    Observable z_i zbar_j w_j wbar_i

    Parameters:
    -----------
    phi: torch.tensor
        Batch of real fields (vectors (...,2n+1,1))

    i: int
        Component z_i
    """

    z, zbar = real2cmplx(phi[:,0]) 
    w, wbar = real2cmplx(phi[:,1])

    O = (z[...,i,:]*zbar[...,j,:]*w[...,j,:]*wbar[...,i,:]).squeeze(-1) # (samples,)

    assert len(O.shape) == 1

    return O