import sys
import os
sys.path.append("src")

import torch
from src.mcmc import * 
from src.deformations import *
from src.model import *
from src.losses import *
from src.observables import *
from src.utils import *

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import datetime
import argparse
import logging
from tabulate import tabulate

def setup_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path + "run.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs', help='observable: ToyOnePt | ToyTwoPt', type=str, default='ToyOnePt')
    parser.add_argument('--i', help='z component', type=int, default=0)
    parser.add_argument('--j', help='z* component', type=int, default=0)
    parser.add_argument('--particle', help='0 => z, 1 => w', type=int, default=0)
    parser.add_argument('--tag', help='tag for saving (one-pt | two-pt | fuzzy-one)', type=str, required=True)
    parser.add_argument('--deformation', help='type of deformation: Linear | Homogeneous', type=str, default='Homogeneous')
    parser.add_argument('--epochs', help='epochs', type=int, default=1_000)
    parser.add_argument('--loss_fn', help='loss function', type=str, default='rlogloss')
    parser.add_argument('--batch_size', help='batch size', type=int, default=1024)
    parser.add_argument('--load_samples', help='which samples to load, those created sequentuially (seq) or with parallel (II) metropolis updates', type=str, default='II')
    args = parser.parse_args()

    assert args.obs in ['ToyOnePt','ToyTwoPt','ToyFullTwoPt','ToyFuzzyOne'], "Wrong observable, please specify one of ['ToyOnePt','ToyTwoPt','ToyFullTwoPt','ToyFuzzyOne']"
    assert args.tag in ['one-pt', 'two-pt', 'fuzzy-one'], "Wrong tag: please specify one of 'one-pt' or 'two-pt'"

    if args.obs == 'ToyOnePt':
        assert args.tag == 'one-pt', f"Incompatible arguments: --obs={args.obs} and --tag={args.tag}"

    if args.obs == 'ToyTwoPt' or args.obs == 'ToyFullTwoPt':
        assert args.tag == 'two-pt', f"Incompatible arguments: --obs={args.obs} and --tag={args.tag}"
    if args.obs == 'ToyFuzzyOne':
        assert args.tag == 'fuzzy-one', f"Incompatible arguments: --obs={args.obs} and --tag={args.tag}" 
    

    ################ SETUP DEVICE  ########################
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

    ################ TOY MODEL ########################
    n = 2
    beta = 4.5

    # ACTION FUNCTIONAL
    S = ToyActionFunctional(n,beta).action

    # OBSERVABLE
    i = args.i 
    j = args.j

    if args.obs == 'ToyOnePt':
        particle = args.particle
        obs = ToyOnePt(i,j,particle)
        obs.name = f"$\\langle O_{{ {i}{j} }}(z)\\rangle$, $\\beta$ = {beta:.1f}"
    if args.obs == 'ToyTwoPt':
        obs = ToyTwoPt(i,j)
        obs.name = f"$\\langle O_{{{i}{j}}}(z)O^\\dagger_{{{i}{j}}}(w)\\rangle$, $\\beta$ = {beta:.1f}"
    if args.obs == 'ToyFullTwoPt':
        obs = ToyFullTwoPt()
        obs.name = f"$\\langle |z* w|^2 \\rangle$, $\\beta$ = {beta:.1f}"
    if args.obs == 'ToyFuzzyOne':
        obs = ToyFuzzyOne()
        obs.name = f"$\\langle 1 \\rangle$, $\\beta$ = {beta:.1f}"

    # DEFORMATION
    deformation_type = args.deformation

    # linear
    if deformation_type == 'Linear':
        a0 = 0.1*torch.randn(phi[0].shape)
        deformation = Linear(a0,n)

    if deformation_type == 'Homogeneous':
        # homogeneous: su(n+1)
        dim_g = n**2 + 2*n
        if args.obs == 'ToyFuzzyOne':
            a0 = 0.1*torch.randn(2,dim_g)
        else:
            a0 = torch.zeros(2,dim_g) 
        deformation = Homogeneous(a0,n)

    batch_size = args.batch_size

    ################ TRAINING ########################
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
        ("action", S),
        ("deformation", deformation),
        ("beta", beta)
    ])
    model = CP(n,**params)

    if torch.cuda.is_available():
        ddp_model = DDP(model.to(device_id),device_ids=[device_id])
    else:
        ddp_model = DDP(model)

    # SET EPOCHS
    epochs = args.epochs

    # TRAINING
    mode = args.load_samples
    print(f"rank {rank}: reading samples..\n")
    phi = torch.load(f'./data/samples_n{n}_b{beta:.1f}_m{mode}.dat',weights_only=True)

    print(f"rank {rank}: starting training..\n")

    observable, observable_var, losses_train, losses_val, anorm, af = train(ddp_model, model, obs, phi, epochs, loss_fn, batch_size=batch_size,lr=lr)
    
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
                ["device", device],
                ["beta (coupling cst)", beta],
                ["n (dimC CP)", n],
                ["dim_g", n**2 + 2*n],
                ["lr (learning rate)", lr],
                ["batch size", batch_size],
                ["loss_fn", args.loss_fn],
                ["epochs", epochs],
                ["obs", args.obs],
                ["(i,j)", (i,j)]
            ]
        try:
            log_data.append(["SLURM_JOB_ID", os.environ['SLURM_JOB_ID']])
        except:
            pass

        table = tabulate(log_data, headers=["param", "value"], tablefmt="grid")

        logger.info("Used Parameters\n\n" + table + "\n")

        plot_params = dict([
            ("n", n),
            ("observable", observable),
            ("observable_var", observable_var),
            ("undeformed_obs", undeformed_obs),
            ("deformed_obs", deformed_obs),
            ("af", af),
            ("anorm", anorm),
            ("losses_train", losses_train),
            ("losses_val", losses_val),
            ("loss_name", args.loss_fn),
            ("deformation_type", deformation_type),
            ("title", obs.name),
            ("path", path)
        ])
        
        save_plots(**plot_params)
        save_tensor(observable, path_raw_data, "observable.pt")
        save_tensor(losses_train, path_raw_data, "losses_train.pt")
        save_tensor(losses_val, path_raw_data, "losses_val.pt")
        save_tensor(af, path_raw_data, "af.pt")
        torch.save(model, path_raw_data + "model.pt") 
        # plot_data(**plot_params)


if __name__ == "__main__":
    # generate_toy_samples(n=2,beta=4.5,N_steps=50_000,mode=args.load_samples)
    main()
