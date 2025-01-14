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
                ee = np.zeros((self.n,self.n))
                ee[j][k] = 1
                sym.append(torch.tensor(ee + ee.transpose()))
                anti_sym.append(torch.tensor(-1j*(ee - ee.transpose())))
                
        for ell in range(self.n-1):
            ee = np.zeros((self.n,self.n))
            ee[ell+1][ell+1] = 1

            d = np.zeros((self.n,self.n)).copy()

            for j in range(ell+1):
                d[j][j] += 1

            b = np.sqrt(2/((ell+1)*((ell+1)+1))) * (d - (ell+1)*ee)

            diag.append(torch.tensor(b))

        return torch.stack(diag + sym + anti_sym,dim=0)

    def embed(self,a):
        x = torch.einsum('...i,ikl->...kl',a.cdouble(),self.basis)

        return x
    
    def rnd_su(self):
        """
        Creates a random element in su(n).
        """

        x = torch.randn(self.dim)

        return self.embed(x)

