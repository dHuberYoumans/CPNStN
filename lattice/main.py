import sys
import os
sys.path.append("../src")
import datetime

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
    print(f"Start running DDP on rank {rank}")

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
    print(f"rank {rank}: reading samples...")
    ens = np.fromfile(f'./data/cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)
    # print("...done\n")

    # SAMPLES
    n=2
    dim_g = n**2 + 2*n

    print(f"rank {rank}: preparing samples...")
    phi = cmplx2real(torch.tensor(ens).unsqueeze(-1).to(device))
    # print("...done\n")
    # print(f"{phi.shape = }")
    # print(f"{phi.dtype = }")

    # ACTION FUNCTIONAL
    S = LatticeActionFunctional(n,beta).action

    # OBSERVABLE
    i, j = 0, 1 
    p = (0,0) # lattice point

    # obs = LatObs.fuzzy_one
    # obs.__name__ = "fuzzy one"

    obs = LatOnePt(p,i,j) # fuzzy_zero
    obs.name = f"$O_{{{i}{j}}}${str(p).replace(' ','')}, $\\beta$ = {beta:.1f}"

    
    a0 = torch.zeros(L,L,dim_g) #1e-8 * torch.rand(L,L,dim)
    # a0[0,0] = 0.1*torch.randn(dim_g)
    # deformation = Homogeneous(a0,n,spacetime="2D")
    lattice_mask = torch.zeros(dim_g,L,L)
    lattice_mask[:,i,j] = 1
    unet = UNET(n,lattice_mask)

    deformation = NNHom(unet,n,spacetime="2D")

    deformation_type = "Lattice"

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
    epochs = 75_000

    # TRAINING
    # print("\n training model ... \n")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(ddp_model,model,phi,epochs=epochs,loss_fct=loss_fct,batch_size=batch_size,lr=lr)
    
    dist.destroy_process_group()

    # print("\n done.\n")

    undeformed_obs = obs(phi)
    deformed_obs = model.Otilde(phi)

    if rank == 0:        # SETUP
        ts = datetime.datetime.today().strftime('%Y.%m.%d_%H:%M')
        path = os.path.join("./plots/",ts + "/")
        path_raw_data = path + "raw_data/"
        os.makedirs(path_raw_data)

        plot_params = dict(
            n = n,
            observable = observable,
            observable_var = observable_var,
            undeformed_obs = undeformed_obs,
            deformed_obs = deformed_obs,
            a0 = a0,
            af = af,
            anorm = anorm,
            losses_train = losses_train,
            losses_val = losses_val,
            loss_name = loss_fct.__name__,
            deformation_type = deformation_type,
            title = obs.name,
            path = path
        )

        # SAVE DATA
        save_plots(**plot_params)
        save_tensor(observable,path_raw_data, "observable.pt")
        save_tensor(losses_train,path_raw_data, "losses_train.pt")
        save_tensor(losses_val,path_raw_data, "losses_val.pt")
        save_tensor(af,path_raw_data, "af.pt")
        torch.save(model,path_raw_data + "model.pt") # save model
        # plot_data(**plot_params)


if __name__ == "__main__":
    mode = ("lattice",)
    main(mode)
