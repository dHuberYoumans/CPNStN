import sys
import os
sys.path.append("src")

import torch
import numpy as np
from src.deformations import *
from src.model import *
from src.losses import *
from src.observables import *
from src.utils import *
from src.unet import UNET

import argparse
import logging
from tabulate import tabulate


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def setup_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path + "run.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs', help='observable: LatOnePt | LatTwoPt', type=str, default='LatOnePt')
    parser.add_argument('--p', help='point', type=tuple_type, default='(0,0)')
    parser.add_argument('--q', help='point', type=tuple_type, default='(0,0)')
    parser.add_argument('--i', help='component', type=int, default=0)
    parser.add_argument('--j', help='component', type=int, default=0)
    parser.add_argument('--k', help='component', type=int, default=0)
    parser.add_argument('--ell', help='component', type=int, default=0)
    parser.add_argument('--tag', help='tag for saving (one-pt | two-pt)', type=str, required=True)
    parser.add_argument('--epochs', help='epochs', type=int, default=1_000)
    parser.add_argument('--loss_fn', help='loss function', type=str, default='rlogloss')
    parser.add_argument('--batch_size', help='batch size', type=int, default=1024)
    args = parser.parse_args()

    assert args.obs == 'LatOnePt' or args.obs == 'LatTwoPt', "Wrong observable, please specify one of 'LatOnePt' or 'LatTwoPt'"
    assert args.tag == 'one-pt' or args.tag == 'two-pt', "Wrong tag: please specify one of 'one-pt' or 'two-pt'"

    if args.obs == 'LatOnePt':
        assert args.tag == 'one-pt', f"Incompatible arguments: --obs={args.obs} and --tag={args.tag}"

    if args.obs == 'LatTwoPt':
        assert args.tag == 'two-pt', f"Incompatible arguments: --obs={args.obs} and --tag={args.tag}"
    
    ################ SETUP DEVICE  ########################

    if torch.cuda.is_available():
        dist.init_process_group("nccl")
    else:
        print("cuda not available. setting backend to gloo\n")
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

    # SAMPLES
    n=2
    dim_g = n**2 + 2*n
 

    # ACTION FUNCTIONAL
    S = LatticeActionFunctional(n,beta).action

    # OBSERVABLE
    i = args.i
    j = args.j
    k = args.k
    ell = args.ell
    p = args.p # lattice point
    q = args.q # lattice point


    if args.obs == 'LatOnePt':
        obs = LatOnePt(p,i,j, n, L) # fuzzy_zero
        obs.name = f"$\\langle O_{{{i}{j}}}${str(p).replace(' ','')}$\\rangle$, $\\beta$ = {beta:.1f}"
    elif args.obs == 'LatTwoPt':
        obs = LatTwoPt(p,q,i,j,k,ell, n, L)
        obs.name = f"$\\langle O_{{{i}{j}}}${str(p).replace(' ','')}$O^\\dagger_{{{k}{ell}}}${str(q).replace(' ','')}$\\rangle$, $\\beta$ = {beta:.1f}"
    
    unet = UNET(n)
    deformation = NNHom(unet,n,spacetime="2D")
    deformation_type = "Lattice"

    batch_size = args.batch_size

    # LEARNING RATE 
    lr = 1e-5

    # LOSS
    if args.loss_fn == 'logloss':
        loss_fn = logloss
    if args.loss_fn == 'rlogloss':
        loss_fn = rlogloss
    if args.loss_fn == 'ilogloss':
        loss_fn = ilogloss
    if args.loss_fn == 'rloss':
        loss_fn = rloss
    if args.loss_fn == 'iloss':
        loss_fn = iloss

    # MODEL
    params = dict([
        ('action', S),
        ('deformation', deformation),
        ('beta', beta)
        ])

    model = CP(n,**params)

    # SET EPOCHS
    epochs = args.epochs 

    
    ################ TRAINING ########################
    if torch.cuda.is_available():
        ddp_model = DDP(model.to(device_id),device_ids=[device_id])
    else:
        ddp_model = DDP(model)

    print(f"rank {rank}: reading samples..\n")
    ens = np.fromfile(f'./data/cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)

    print(f"rank {rank}: preparing samples..\n")
    phi = cmplx2real(torch.tensor(ens).unsqueeze(-1).to(device))

    print(f"rank {rank}: starting training..\n")

    training_param = dict([
        ('ddp_model', ddp_model),
        ('model', model),
        ('obs', obs),
        ('phi', phi),
        ('epochs', epochs),
        ('loss_fn', loss_fn),
        ('lr', lr),
        ('batch_size', batch_size)
        ])

    observable, observable_var, losses_train, losses_val, anorm, af = train(**training_param)
    
    print(f"rank {rank}: ...finished.\n")

    dist.destroy_process_group()


    undeformed_obs = obs(phi)
    deformed_obs = model.Otilde(obs, phi)

    if rank == 0: 
        ts = datetime.datetime.today().strftime('%Y.%m.%d_%H:%M')
        path = os.path.join(f"./plots/{args.tag}/",ts + "/")
        path_raw_data = path + "raw_data/"

        try:
            os.makedirs(path_raw_data)
        except FileExistsError:
            pass

        logger = setup_logger(path)

        print("Saving data..\n")

        # LOGGING PARAMETERS
        log_data = [
                ["device",device],
                ["L (lattice size)",L],
                ["beta (coupling cst)", beta],
                ["n (dimC CP)", n],
                ["dim_g",dim_g],
                ["lr (learning rate)", lr],
                ["batch size", batch_size],
                ["loss_fn",loss_fn.__name__],
                ["epochs",epochs],
                ["obs", args.obs],
            ]
        if args.obs == "LatOnePt":
            log_data.append(["p",p])
            log_data.append(["(i,j)",(i,j)])
        elif args.obs == "LatTwoPt":
            log_data.append(["(p,q)",(p,q)])
            log_data.append(["(i,j)",(i,j)])
            log_data.append(["(k,l)",(k,ell)])
        try:
            log_data.append(["SLURM_JOB_ID", os.environ['SLURM_JOB_ID']])
        except:
            pass

        table = tabulate(log_data, headers=["param", "value"], tablefmt="grid")

        logger.info("Used Parameters\n\n" + table + "\n")

        plot_params = dict([
            ('n' , n),
            ('observable' , observable),
            ('observable_var' , observable_var),
            ('undeformed_obs' , undeformed_obs),
            ('deformed_obs' , deformed_obs),
            ('af' , af),
            ('anorm' , anorm),
            ('losses_train' , losses_train),
            ('losses_val' , losses_val),
            ('loss_name' , loss_fn.__name__),
            ('deformation_type' , deformation_type),
            ('title' , obs.name),
            ('path' , path)
        ])

        # SAVE DATA
        save_plots(**plot_params)
        save_tensor(observable,path_raw_data, "observable.pt")
        save_tensor(losses_train,path_raw_data, "losses_train.pt")
        save_tensor(losses_val,path_raw_data, "losses_val.pt")
        save_tensor(af,path_raw_data, "af.pt")
        torch.save(model,path_raw_data + "model.pt")
        # plot_data(**plot_params)


if __name__ == "__main__":
    main()
