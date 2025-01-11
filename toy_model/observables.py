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

    Z = phi[:,0]

    # COMPLEX VARIABLES
    z, zbar = real2cmplx(Z)

    # COMPLEX OBSERVABLE
    O = (z[...,i,:]*zbar[...,j,:]) 

    return O

def two_pt(phi,i):
    """
    Observable |z_i|^2

    Parameters:
    -----------
    phi: torch.tensor
        Batch of real fields (vectors (...,2n+1,1))

    i: int
        Component z_i
    """

    Z = phi[:,0] # real rep of z
    # W = phi[:,1]

    # COMPLEX VARIABLES
    z, zbar = real2cmplx(Z) # (*batch,dim_C,1) 
    # w, wbar = real2cmplx(W) # (*batch,dim_C,1) 

    # COMPLEX OBSERVABLE
    O = (z[...,i,:]*zbar[...,i,:]) # z_i \bar z_j (*batch,,1)

    return O