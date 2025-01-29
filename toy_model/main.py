import torch
from mcmc import * 
from deformations import *
from model import *
from losses import *
from observables import *
from utils import grab

import analysis as al

import matplotlib.pyplot as plt
import seaborn as sns

def main(mode):
    if mode[0] == "lattice":
    ################ LATTICE ########################
        L = 64
        beta = 4.0
        Nc = 3
        n_cfg = 1_000
        print("Reading samples..\n")
        ens = np.fromfile(f'cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)
        print("...done\n")

        # SAMPLES
        n=2
        rk = n
        dim_su = n**2 + 2*n

        print("Preparing samples..\n")
        phi = cmplx2real(torch.tensor(ens).unsqueeze(-1))
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
        deformation = Homogeneous(a0,n,spacetime="2D")

        deformation_type = "lattice"

        batch_size = 64

    if mode[0] == "toy":
        ################ TOY MODEL ########################
        n = 2
        beta = 4.5

        # SAMPLES
        phi = torch.load(f'samples_n{n}_b{beta:.1f}_m{mode[1]}.dat',weights_only=True)

        # ACTION FUNCTIONAL
        S = lambda phi: ToyActionFunctional(n).action(phi,beta)

        # OBSERVABLE
        i,j = 0, 1

        # obs = ToyObs.fuzzy_one

        # obs = lambda phi: ToyObs.one_pt(phi,i,j) # fuzzy_zero
        # obs.__name__ = "$O_{{ {i}{j} }}$(z), $\\beta$ = {beta:1.f}"

        obs = lambda phi: ToyObs.two_pt(phi,i,j)
        obs.__name__ = f"$O_{{{i}{j}}}$(z,w), $\\beta$ = {beta:.1f}"

        # obs = lambda phi: ToyObs.two_pt_full(phi)
        # obs.__name__ = f"$\langle |z^\dagger w|^2 \\rangle$, $\\beta$ = {beta:.1f}"



        # DEFORMATION

        # linear
        # a0 = 0.1*torch.randn(phi[0].shape) # dim(a) = 2n + 2
        # deformation = Linear(a0,n)

        # homogeneous: su(n+1)
        rk = n
        dim = n**2 + 2*n
        a0 = torch.zeros(2,dim) 
        # a0 = 0.1*torch.randn(2,dim) # fuzzy one 
        deformation = Homogeneous(a0,n)


        deformation_type = deformation.__class__.__name__
        batch_size = 1024

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

    # SET EPOCHS
    epochs = 10_000

    # TRAINING
    print("\n training model ... \n")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(model,phi,epochs=epochs,loss_fct=loss_fct,batch_size=batch_size,lr=lr)

    print("\n done.\n")

    undeformed_obs = obs(phi)
    deformed_obs = model.Otilde(phi)

    plot_comparison(undeformed_obs,deformed_obs)
    plot_data(n,observable,observable_var,undeformed_obs,af,anorm,losses_train,losses_val,loss_fct.__name__,deformation_type,obs.__name__)#loss_name[loss_fct]

def plot_data(n,observable,observable_var,undeformed_obs,af,anorm,losses_train,losses_val,loss_name,deformation_type,title=None):
    # VARIANCE PLOT
    epochs = len(observable)
    

    Nboot = 1_000
    mean_re, err_re = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_im, err_im = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)


    # LINEAR DEFORMATION
    if deformation_type == "Linear":
        aZ = torch.tensor(af[0]).cfloat()
        aW = torch.tensor(af[1]).cfloat()
    elif deformation_type == "Homogeneous":
    # HOMOGENEOUS DEFORMATION
        su_n = LieSU(n+1)
        aZ = su_n.embed(torch.tensor(af[0]))
        aW = su_n.embed(torch.tensor(af[1]))
    else:
        su_n = LieSU(n+1)
        aZ = su_n.embed(torch.tensor(af[0,0]))
        aW = su_n.embed(torch.tensor(af[0,1]))


    # OBSERVABLE
    fig, ax = plt.subplots(nrows=2,ncols=2) #[1, 1, 2, 2] gridspec_kw={'height_ratios': [1,1]}

    ax[0,1].plot(anorm)
    ax[0,1].set_title('norm a')

    ax[0,0].plot(*np.transpose(losses_train),label='loss')
    ax[0,0].plot(*np.transpose(losses_val),label='val_loss')
    ax[0,0].legend()
    ax[0,0].set_title(loss_name)
    ax[0,0].legend()

    ax[1,0].plot(*np.transpose([(e,z.real) for e,z in observable]),label='re')
    # ax[1,0].plot([z.imag for z in observable],label='im',color='purple')
    # ax[1,0].plot([z for z in observable],label='re') # full 2pt fct
    ax[1,0].axhline(y=mean_re,xmin=0,xmax=epochs,label='OG re',color='red')
    # ax[1,0].axhline(y=mean_im,xmin=0,xmax=epochs,label='OG im',color='orange')
    ax[1,0].fill_between([-100,epochs], [mean_re-err_re]*2, [mean_re+err_re]*2, alpha=0.5, color='red')
    # ax[1,0].fill_between([-100,epochs], [mean_im-err_im]*2, [mean_im+err_im]*2, alpha=0.5, color='orange')
    ax[1,0].set_title('defromed obs')
    ax[1,0].legend()

    ax[1,1].plot(*np.transpose([(e,z.real) for e,z in observable_var]),label='deformed')
    ax[1,1].axhline(y=undeformed_obs.var(),xmin=0,xmax=epochs,label='undeformed',color='red')
    ax[1,1].set_title('obs variance')
    ax[1,1].legend()

    plt.suptitle(title)
    plt.tight_layout();

    # LEARNED DEFORMATION PARAMETER
    fig, ax = plt.subplots(nrows=2,ncols=2)

    sns.heatmap(data=aZ.real,ax=ax[0,0],cmap='coolwarm')
    ax[0,0].set_title('real(a[0]) after training')
    sns.heatmap(data=aZ.imag,ax=ax[0,1],cmap='coolwarm')
    ax[0,1].set_title('imag(a[0]) after training')

    sns.heatmap(data=aW.real,ax=ax[1,0],cmap='coolwarm')
    ax[1,0].set_title('real(a[1]) after training')
    sns.heatmap(data=aW.imag,ax=ax[1,1],cmap='coolwarm')
    ax[1,1].set_title('imag(a[1]) after training')

    plt.suptitle(title + " deformation params")
    plt.tight_layout()
    plt.show();

def generate_toy_samples(n,beta,N_steps = 10_000, burnin = 1_000, skip = 10, mode = 'II'):
    """ GENERATING SAMPLES """

    phi0 = torch.randn(2,(2*n + 2),1).double()
    phi0 /= torch.linalg.vector_norm(phi0, dim=1, keepdim=True)

    print(f"creating samples ({mode = })... \n")

    fn_map = {
            'II': create_samples_II,
            'seq': create_samples_seq
    }

    phi, alpha = fn_map[mode](n=n,phi0=phi0,beta=beta,N_steps=N_steps,burnin=burnin,k=skip)

    print("\ndone")
    print(f"\n{phi.shape = }\t{alpha = }\n")
    print("\nsaving ...")

    torch.save(phi, f"samples_n{n}_b{beta}_m{mode}.dat")

    print("\ndone")

def plot_comparison(undeformed_obs, deformed_obs):
    

    Nboot = 1000
    mean_re_og, err_re_og = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_im_og, err_im_og = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)
    mean_re, err_re = al.bootstrap(grab(deformed_obs),Nboot=Nboot,f=al.rmean)
    mean_im, err_im = al.bootstrap(grab(deformed_obs),Nboot=Nboot,f=al.imean)

    print(f"ratio err bars: {err_re_og / err_re}")

    plt.errorbar([0],[mean_re_og],[err_re_og],color='blue',label='OG',marker='o',capsize=2)
    plt.errorbar([1],[mean_re],[err_re],color='red',label='def',marker='o',capsize=2)
    plt.xlim(-1,2)
    plt.title("errorbar comparison before and after training")
    plt.legend();
    # plt.show();


if __name__ == "__main__":
    # mode = ("toy","II")
    # generate_toy_samples(n=2,beta=4.5,N_steps=50_000,mode=mode)
    mode = ("lattice",)
    main(mode)
