import torch
import numpy as np

def rho(z):
    """
    Real representation of Mat(n x n,C) in terms of Mat( 2n x 2n, R) using z = a + i b => [[a,-b],[b,a]] for any complex entry z 
    """
    assert z.shape[-2] == z.shape[-1], f'non-batch dim of z must be the same. Found {tuple(z.shape[-2:])}'

    *batch_dims, n, m = z.shape
    double_last_dim = (2*n, 2*m)
    
    z_ = torch.empty(*batch_dims,*double_last_dim)
    
    re = z.real
    im = z.imag

    z_[...,0::2,0::2] = re
    z_[...,0::2,1::2] = -im
    z_[...,1::2,0::2] = im
    z_[...,1::2,1::2] = re

    return z_

def inner(X,Y):
    assert X.shape[-1] == 1 and Y.shape[-1] == 1, "Expect column vectors"

    return torch.sum((X*Y).squeeze(-1),dim=-1)

def real2cmplx(phi):
    z = phi[...,::2,:] + 1j*phi[...,1::2,:]
    zbar = phi[...,::2,:] - 1j*phi[...,1::2,:]

    return (z.cdouble(),zbar.cdouble())
    
def cmplx2real(z):
    assert z.is_complex(), "Input tensor must be complex"

    dtype = z.real.dtype

    re = z.real
    im = z.imag
    Xshape = list(z.shape) 
    Xshape[-2] *= 2

    X = torch.zeros(*Xshape,dtype=dtype)
    X[...,::2,:] = re
    X[...,1::2,:] = im

    return X
    
class LieSU():
    def __init__(self,n):
        self.n = n
        self.dim = n**2 - 1
        self.rank = n - 1
        self.basis = self.compute_basis()

    def compute_basis(self):
        """
        Computes a basis for su(n) generalizing Pauli and Gell-Mann matrices. Generators of su(n) are taken to be Hermitian.
        """
        sym = []
        anti_sym = []
        diag = []

        for j in range(self.n):
            for k in range(j+1,self.n):
                ee = torch.zeros((self.n,self.n),dtype=torch.cdouble)
                ee[j][k] = 1
                sym.append(ee + ee.transpose(-1,-2))
                anti_sym.append(-1j*(ee - ee.transpose(-1,-2)))

        for ell in range(self.n - 1): # easier but not orthogonal basis! 
            ee = torch.zeros((self.n,self.n),dtype=torch.cdouble)
            ee[ell][ell] = 1
            ee[ell+1][ell+1] = -1

            diag.append(ee)
                
        # for ell in range(self.n-1):
        #     ee = np.zeros((self.n,self.n))
        #     ee[ell+1][ell+1] = 1

        #     d = np.zeros((self.n,self.n)).copy()

        #     for j in range(ell+1):
        #         d[j][j] += 1

        #     b = np.sqrt(2/((ell+1)*((ell+1)+1))) * (d - (ell+1)*ee)

        #     diag.append(torch.tensor(b))

        return torch.stack((*diag,*sym,*anti_sym),dim=0)

    def embed(self,a):
        x = torch.einsum('...i,ikl->...kl',a.cdouble(),self.basis)

        return x
    
    def rnd_su(self):
        """
        Creates a random element in su(n).
        """

        x = torch.randn(self.dim)

        return self.embed(x)

