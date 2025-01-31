import sys
sys.path.append("../src")

import torch
import numpy as np
from mcmc import * 
from deformations import *
from model import *
from losses import *
from observables import *
from utils import *
from unet import UNET


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main(mode):
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
    else:
        print("cuda not available. setting backend to gloo")
        dist.init_process_group("gloo")

    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}\n")

    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = torch.device("cuda:" + str(device_id)) 
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)

    ################ LATTICE ########################
    L = 64
    beta = 4.0
    Nc = 3
    n_cfg = 1_000
    print("Reading samples..\n")
    ens = np.fromfile(f'./data/cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)
    print("...done\n")

    # SAMPLES
    n=2
    dim_su = n**2 + 2*n

    print("Preparing samples..\n")
    phi = cmplx2real(torch.tensor(ens).unsqueeze(-1).to(device))
    print("...done\n")
    print(f"{phi.shape = }")
    print(f"{phi.dtype = }")

    # ACTION FUNCTIONAL
    S = lambda phi: LatticeActionFunctional(n).action(phi.cdouble(),beta)

    # OBSERVABLE
    i, j = 0, 1 
    p = (0, 0) # lattice point

    # obs = LatObs.fuzzy_one
    # obs.__name__ = "fuzzy one"

    obs = lambda phi: LatObs.one_pt(phi,p,i,j) # fuzzy_zero
    obs.__name__ = f"$O_{{{i}{j}}}${p}, $\\beta$ = {beta:.1f}"

    # obs = lambda phi: LatObs.two_pt(phi,i,j)
    
    
    a0 = torch.zeros(L,L,dim_su) #1e-8 * torch.rand(L,L,dim)
    # a0[0,0] = 0.1*torch.randn(dim_su)
    # deformation = Homogeneous(a0,n,spacetime="2D")
    unet = UNET(n,L,L)

    deformation = NNHom(unet,n,spacetime="2D")

    deformation_type = "lattice"

    batch_size = 128

    ################ TRAINING ########################
    # LEARNING RATE 
    lr = 1e-5

    # LOSS
    loss_fct = rlogloss

    # MODEL
    params = dict(
        action = S,
        deformation = deformation,
        observable = obs,
        beta = beta
    )

    model = CP(n,**params)

    if torch.cuda.is_available():
        ddp_model = DDP(model.to(device_id),device_ids=[device_id])
    else:
        ddp_model = DDP(model)

    # SET EPOCHS
    epochs = 10

    # TRAINING
    print("\n training model ... \n")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(ddp_model,model,phi,epochs=epochs,loss_fct=loss_fct,batch_size=batch_size,lr=lr)
    
    dist.destroy_process_group()

    print("\n done.\n")

    undeformed_obs = obs(phi)
    deformed_obs = model.Otilde(phi)

    if rank == 0:
        plot_params = dict(
            n = n,
            observable = observable,
            observable_var = observable_var,
            undeformed_obs = undeformed_obs,
            deformed_obs = deformed_obs,
            af = af,
            anorm = anorm,
            losses_train = losses_train,
            losses_val = losses_val,
            loss_name = loss_fct.__name__,
            deformation_type = deformation_type,
            title = obs.__name__
        )
        
        # save_plots(**plot_params)
        plot_data(**plot_params)


if __name__ == "__main__":
    mode = ("lattice",)
    main(mode)
