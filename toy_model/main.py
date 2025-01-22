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
    if mode == "lattice":
    ################ LATTICE ########################
        L = 64
        beta = 4.5
        Nc = 3
        n_cfg = 1_000
        print("Reading samples..\n")
        ens = np.fromfile(f'cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)
        print("...done\n")

        # SAMPLES
        n=2
        rk = n
        dim = n**2 + 2*n

        print("Preparing samples..\n")
        phi = cmplx2real(torch.tensor(ens).unsqueeze(-1))
        print("...done\n")
        print(f"{phi.shape = }")
        print(f"{phi.dtype = }")

        # ACTION FUNCTIONAL
        S = lambda phi: LatticeActionFunctional(n).action(phi.cdouble(),beta)

        # OBSERVABLE
        i,j = 0, 1 
        obs = LatObs.fuzzy_one
        # obs = lambda phi: LatObs.one_pt(phi,i,j) # fuzzy_zero
        # obs = lambda phi: LatObs.two_pt(phi,i,j)
        
        a0 = 0.001 * torch.rand(L,L,dim)
        deformation = Homogeneous(a0,n,mode="2D")

        deformation_type = "lattice"

        batch_size = 16

    if mode == "toy":
        ################ TOY MODEL ########################
        n = 2
        beta = 1.0

        # SAMPLES
        phi = torch.load(f'samples_n{n}_b{beta:.1f}.dat',weights_only=True)

        # ACTION FUNCTIONAL
        S = lambda phi: ToyActionFunctional(n).action(phi,beta)

        # OBSERVABLE
        i,j = 0, 1
        # obs = ToyObs.fuzzy_one
        # obs = lambda phi: ToyObs.one_pt(phi,i,j) # fuzzy_zero
        obs = lambda phi: ToyObs.two_pt(phi,i,j)

        # deformation_type = "linear"
        # a0 = 0.1*torch.randn(phi[0].shape) # dim(a) = 2n + 2
        # deformation = Linear(a0,n)

        # su(n+1)
        deformation_type = "homogeneous"
        rk = n
        dim = n**2 + 2*n
        a0 = torch.zeros(2,dim) 
        # a0 = 0.1*torch.randn(2,dim) # fuzzy one
        # a0 = torch.stack([torch.cat([0.1*torch.randn(rk),torch.zeros(dim-rk)]),torch.cat([0.1*torch.randn(rk),torch.zeros(dim-rk)])],dim=0) # torus
        deformation = Homogeneous(a0,n)

        batch_size = 1024

    ################ TRAINING ########################
    # LEARNING RATE 
    alpha = 1e-3

    # LOSS
    loss_fct = loss
    loss_name = 'loss' if loss_fct == loss else 'logloss'

    # MODEL
    params = dict(
        action = S,
        deformation = deformation,
        observable = obs,
        beta = beta
    )
    model = CP(n,**params)

    # SET EPOCHS
    epochs = 5_000

    # TRAINING
    print("\n training model ... \n")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(model,phi,epochs=epochs,loss_fct=loss_fct,batch_size=batch_size,lr=alpha)

    print("\n done.\n")

    undeformed_obs = obs(phi)

    print(f"trace of emebedding: {LieSU(n+1).embed(torch.tensor(af[0])).trace()}\n")

    plot_data(n,observable,observable_var,undeformed_obs,af,anorm,losses_train,losses_val,loss_name,deformation_type)

def plot_data(n,observable,observable_var,undeformed_obs,af,anorm,losses_train,losses_val,loss_name,deformation_type):
    # VARIANCE PLOT
    epochs = len(observable)
    

    Nboot = 1000
    mean_re, err_re = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_im, err_im = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)


    # LINEAR DEFORMATION
    if deformation_type == "linear":
        aZ = torch.tensor(af[0]).cfloat()
        aW = torch.tensor(af[1]).cfloat()
    elif deformation_type == "homogeneous":
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

    ax[0,0].plot(losses_train,label='loss')
    ax[0,0].plot(losses_val,label='val_loss')
    ax[0,0].legend()
    ax[0,0].set_title(loss_name)
    ax[0,0].legend()

    ax[1,0].plot([z.real for z in observable],label='re')
    ax[1,0].plot([z.imag for z in observable],label='im',color='purple')
    ax[1,0].axhline(y=mean_re,xmin=0,xmax=epochs,label='OG re',color='red')
    ax[1,0].axhline(y=mean_im,xmin=0,xmax=epochs,label='OG im',color='orange')
    ax[1,0].fill_between([-100,epochs], [mean_re-err_re]*2, [mean_re+err_re]*2, alpha=0.5, color='red')
    ax[1,0].fill_between([-100,epochs], [mean_im-err_im]*2, [mean_im+err_im]*2, alpha=0.5, color='orange')
    ax[1,0].set_title('defromed obs')
    ax[1,0].legend()

    ax[1,1].plot([z.real for z in observable_var],label='deformed')
    ax[1,1].axhline(y=undeformed_obs.var(),xmin=0,xmax=epochs,label='undeformed',color='red')
    ax[1,1].set_title('obs variance')
    ax[1,1].legend()

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

    plt.tight_layout()
    plt.show();

def generate_toy_samples(n,beta,N_steps = 10_000, burnin = 1_000, skip = 10):
    """ GENERATING SAMPLES """

    phi0 = torch.randn(2,(2*n + 2),1).double()
    phi0 /= torch.linalg.vector_norm(phi0, dim=1, keepdim=True)

    print("creating samples ... \n")

    phi, alpha = create_samples(n=n,phi0=[phi0],beta=beta,N_steps=N_steps,burnin=burnin,k=skip)

    print("\ndone")
    print(f"\n{phi.shape = }\t{alpha = }\n")
    print("\nsaving ...")

    torch.save(phi, f"samples_n{n}_b{beta}.dat")

    print("\ndone")


if __name__ == "__main__":
    # generate_toy_samples(n=2,beta=1.0)
    main("toy")