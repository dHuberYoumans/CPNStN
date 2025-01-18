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

if __name__ == "__main__":
    ################ GENERATING SAMPLES ################

    # n = 2
    # beta = 1.0

    # N_steps = 25_000
    # burnin = 1_000
    # skip = 10

    # phi0 = torch.randn(2,(2*n + 2),1)
    # phi0 /= torch.linalg.vector_norm(phi0, dim=1, keepdim=True)

    # print("creating samples ... \n")

    # phi, alpha = create_samples(n=n,phi0=phi0,beta=beta,N_steps=N_steps,burnin=burnin,k=skip)

    # print("\ndone")
    # print(f"\n{phi.shape = }\t{alpha = }\n")
    # print("\nsaving ...")

    # torch.save(phi, f"samples_n{n}_b{beta}.dat")

    # print("\ndone")

    # exit()

    ################ SET HYPERPARAMETERS ################
    n = 2
    beta = 1.0

    alpha = 1e-3 # learning rate
    i,j = 0, 1 # parameter for fuzzy zero
    # obs = fuzzy_one
    # obs = lambda phi: one_pt(phi,i,j) # fuzzy_zero
    obs = lambda phi: two_pt(phi,i,j)

    ################ LATTICE ########################
    # Lx = 8
    # Ly = 8
    # phi = 0.1*torch.randn([1000,8,8,6,1])
    # print(f"\n{phi.shape = }\n")

    # lattice = LatticeActionFunctional(n)
    # S = lambda phi: lattice.S(phi.cdouble(),beta)

    # rk = n
    # dim = n**2 + 2*n
    # a0 = 0.1*torch.randn(Lx,Ly,dim)
    # deformation = Homogeneous(a0,n,mode="2D")

    ################ TOY MODEL ########################

    phi = torch.load(f'samples_n{n}_b{beta:.1f}.dat',weights_only=True)

    toy_model = ToyActionFunctional(n)
    S = lambda phi: toy_model.S(phi,beta)

    # deformation_type = "linear"
    # a0 = 0.1*torch.randn(phi[0].shape) # dim(a) = 2n + 2
    # deformation = Linear(a0,n)

    # su(n+1)
    deformation_type = "homogeneous"
    rk = n
    dim = n**2 + 2*n
    a0 = 0.1*torch.randn(2,dim) # full hom
    # a0 = torch.stack([torch.cat([0.1*torch.randn(rk),torch.zeros(dim-rk)]),torch.cat([0.1*torch.randn(rk),torch.zeros(dim-rk)])],dim=0) # torus
    deformation = Homogeneous(a0,n)

    ################ TRAINING ########################

    # LOSS
    loss_fct = loss
    loss_name = 'loss' if loss_fct == loss else 'logloss'

    # MODEL
    params = [S,deformation,obs,beta]
    model = CP(n,*params)

    # SET EPOCHS
    epochs = 5_000

    # TRAINING
    print("\n training model ... \n")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(model,phi,epochs=epochs,loss_fct=loss_fct,batch_size=1024)

    print("\n done.\n")

    # VARIANCE PLOT
    undeformed_obs = obs(phi)

    Nboot = 1000
    mean_re, err_re = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_im, err_im = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)


    # LINEAR DEFORMATION
    if deformation_type == "linear":
        aZ = torch.tensor(af[0]).cfloat()
        aW = torch.tensor(af[1]).cfloat()
    else:
    # HOMOGENEOUS DEFORMATION
        su_n = LieSU(n+1)
        aZ = su_n.embed(torch.tensor(af[0]))
        aW = su_n.embed(torch.tensor(af[1]))

    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(15,10),gridspec_kw={'height_ratios': [1, 1, 2, 2]})

    plt.suptitle('deformation\n')
    ax[0,1].plot(anorm)
    ax[0,1].set_title('norm a')

    ax[0,0].plot(losses_train,label='loss')
    ax[0,0].plot(losses_val,label='val_loss')
    ax[0,0].legend()
    ax[0,0].set_title(loss_name)
    ax[0,0].legend()

    ax[1,0].plot([z.real for z in observable],label='re')
    ax[1,0].plot([z.imag for z in observable],label='im',color='purple')
    ax[1,0].axhline(y=mean_re,xmin=0,xmax=phi.shape[0],label='OG re',color='red')
    ax[1,0].axhline(y=mean_im,xmin=0,xmax=phi.shape[0],label='OG im',color='orange')
    ax[1,0].fill_between([0,epochs], [mean_re-err_re]*2, [mean_re+err_re]*2, alpha=0.5, color='red')
    ax[1,0].fill_between([0,epochs], [mean_im-err_im]*2, [mean_im+err_im]*2, alpha=0.5, color='orange')
    ax[1,0].set_title('defromed obs')
    ax[1,0].legend()

    ax[1,1].plot([z.real for z in observable_var],label='deformed')
    ax[1,1].axhline(y=undeformed_obs.var(),xmin=0,xmax=phi.shape[0],label='undeformed',color='red')
    ax[1,1].set_title('obs variance')
    ax[1,1].legend()

    sns.heatmap(data=aZ.real,ax=ax[2,0],cmap='coolwarm',vmin=-0.1,vmax=0.1)
    ax[2,0].set_title('real(a[0]) after training')
    sns.heatmap(data=aZ.imag,ax=ax[2,1],cmap='coolwarm',vmin=-0.1,vmax=0.1)
    ax[2,1].set_title('imag(a[0]) after training')

    sns.heatmap(data=aW.real,ax=ax[3,0],cmap='coolwarm',vmin=-0.1,vmax=0.1)
    ax[3,0].set_title('real(a[1]) after training')
    sns.heatmap(data=aW.imag,ax=ax[3,1],cmap='coolwarm',vmin=-0.1,vmax=0.1)
    ax[3,1].set_title('imag(a[1]) after training')

    plt.tight_layout()
    plt.show();


