import torch
from mcmc import * 
from deformations import *
from model import *
from losses import *
from observables import *

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

    # torch.save(phi, "samples.dat")

    # print("\ndone")

    # exit()

    ################ LOAD SAMPLES ################

    phi = torch.load('samples.dat',weights_only=True)
    print(f"\n{phi.shape = }\n")

    ################ SET HYPERPARAMETERS ################
    n = 2
    beta = 1.0

    alpha = 1e-4 # learning rate
    i,j = 2, 1 # parameter for fuzzy zero
    # obs = fuzzy_one
    obs = lambda psi: fuzzy_zero(psi,i,j)
    # obs = lambda phi: two_pt(phi,i)

    ################ ACTION ########################
    action = ToyActionFunctional()
    Stoy = lambda psi: action.Stoy(psi,beta)

    ################ DEFORMATION ########################

    # a0 = 0.1*torch.ones_like(phi[0]) # dim(a) = 2n + 2
    # deformation = Linear()

    # su(n+1)
    rk = n
    dim = n**2 + 2*n
    a0 = 0.1*torch.randn(dim) # full hom
    # a0 = torch.cat([0.1*torch.randn(rk),torch.zeros(dim-rk)]) # torus
    deformation = Homogeneous()

    # LOSS
    loss_fct = loss
    loss_name = 'loss' if loss_fct == loss else 'logloss'

    # MODEL
    params = [Stoy,deformation,a0,obs,beta]
    model = CP(n,*params)

    # SET EPOCHS
    epochs = 1_000

    # TRAINING
    print("\n training model ... ")

    observable, observable_var, losses_train, losses_val, anorm, a0, af = train(model,phi,epochs=epochs,loss_fct=loss_fct)

    print("\n done.")

    # VARIANCE PLOT
    undeformed_obs = obs(phi)
    su_n = LieSU(n+1)

    a_init = su_n.embed(a0)
    a_final = su_n.embed(af)

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
    ax[1,0].axhline(y=undeformed_obs.mean().real,xmin=0,xmax=phi.shape[0],label='OG re',color='red')
    ax[1,0].axhline(y=undeformed_obs.mean().imag,xmin=0,xmax=phi.shape[0],label='OG im',color='orange')
    ax[1,0].set_title('defromed obs')
    ax[1,0].legend()

    ax[1,1].plot([z.real for z in observable_var],label='deformed')
    ax[1,1].axhline(y=undeformed_obs.var(),xmin=0,xmax=phi.shape[0],label='undeformed',color='red')
    ax[1,1].set_title('obs variance')
    ax[1,1].legend()

    sns.heatmap(data=a_init.real,ax=ax[2,0],cmap='coolwarm')
    ax[2,0].set_title('real(a) before training')
    sns.heatmap(data=a_init.imag,ax=ax[2,1],cmap='coolwarm')
    ax[2,1].set_title('imag(a) before training')

    sns.heatmap(data=a_final.real,ax=ax[3,0],cmap='coolwarm')
    ax[3,0].set_title('real(a) after training')
    sns.heatmap(data=a_final.imag,ax=ax[3,1],cmap='coolwarm')
    ax[3,1].set_title('imag(a) after training')

    plt.tight_layout()
    plt.show();


