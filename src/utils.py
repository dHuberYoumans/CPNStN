import numpy as np
import time
import torch 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

import analysis as al
from linalg import *

import os
import datetime

def grab(x,safe=True): 
    arr = x.detach().cpu().numpy()

    if safe:
        if np.any(np.isnan(arr)):
            print("Warning: NaN!")

        if np.any(np.isinf(arr)):
            print("Warning: Inf detected in tensor!")

    return arr

class Timer:
    """
    A simple context manager for timing.
    """
    def __init__(self,msg):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self,exc_type, exc_val, exc_tb):
        end_time = time.time()
        t1 = end_time - self.t0
        print(self.msg,f"took {t1:.4f}s")
        return False  


def plot_data(n,observable,observable_var,undeformed_obs,deformed_obs,af,anorm,losses_train,losses_val,loss_name,deformation_type,title=None):
    # VARIANCE PLOT
    epochs = len(observable)
    
    Nboot = 1_000
    mean_re_og, err_re_og = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_re, err_re = al.bootstrap(grab(deformed_obs),Nboot=Nboot,f=al.rmean)
    # mean_im_og, err_im_og = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)
    # mean_im, err_im = al.bootstrap(grab(deformed_obs),Nboot=Nboot,f=al.imean)

    # mean_re, err_re = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    # mean_im, err_im = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.imean)


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
    ax[0,1].set_xlabel("epochs")
    ax[0,1].set_title(r"$\Vert a \Vert$")

    ax[0,0].plot(*np.transpose(losses_train),label="training")
    ax[0,0].plot(*np.transpose(losses_val),label="validation")
    ax[0,0].set_xlabel("epochs")
    ax[0,0].set_title(loss_name)
    ax[0,0].legend()

    ax[1,0].plot(*np.transpose([(e,z.real) for e,z in observable]),label="def")
    ax[1,0].axhline(y=mean_re_og,xmin=0,xmax=epochs,label="og",color='red')
    ax[1,0].fill_between([-100,epochs], [mean_re_og-err_re]*2, [mean_re_og+err_re]*2, alpha=0.5, color='red')
    # ax[1,0].plot([(e,z.imag) for e,z in observable],label='im',color='purple')
    # ax[1,0].axhline(y=mean_im,xmin=0,xmax=epochs,label='OG im',color='orange')
    # ax[1,0].fill_between([-100,epochs], [mean_im-err_im]*2, [mean_im+err_im]*2, alpha=0.5, color='orange')
    # ax[1,0].plot([z for z in observable],label='re') # full 2pt fct
    ax[1,0].set_xlabel("epochs")
    ax[1,0].set_title(r"$\langle$" + title.split(' ')[0][:-1] + r"$\rangle$")
    ax[1,0].legend()

    ax[1,1].plot(*np.transpose([(e,z.real) for e,z in observable_var]),label="def")
    ax[1,1].axhline(y=grab(undeformed_obs.var()),xmin=0,xmax=epochs,label="og",color='red')
    ax[1,1].set_xlabel("epochs")
    ax[1,1].set_title("variance")
    ax[1,1].legend()

    plt.suptitle(title)
    plt.tight_layout();

    # LEARNED DEFORMATION PARAMETER
    fig, ax = plt.subplots(nrows=2,ncols=2)

    sns.heatmap(data=aZ.real,ax=ax[0,0],cmap='coolwarm')
    ax[0,0].set_title(r"${\rm Re}(a(z))$")
    sns.heatmap(data=aZ.imag,ax=ax[0,1],cmap='coolwarm')
    ax[0,1].set_title(r"${\rm Im}(a(z))$")

    sns.heatmap(data=aW.real,ax=ax[1,0],cmap='coolwarm')
    ax[1,0].set_title(r"${\rm Re}(a(w))$")
    sns.heatmap(data=aW.imag,ax=ax[1,1],cmap='coolwarm')
    ax[1,1].set_title(r"${\rm Im}(a(w))$")

    plt.suptitle(title + ", deformation parameters")
    plt.tight_layout()

    # ERRORBARS 
    fig = plt.figure()
    plt.errorbar([0],[mean_re_og],[err_re_og],color='blue',label=r"$\varepsilon_{og}$",marker='o',capsize=2)
    plt.errorbar([1],[mean_re],[err_re],color='red',label=r"$\varepsilon_{def}$",marker='o',capsize=2)
    plt.xlim(-1,2)
    plt.xticks([],[])
    plt.title(f"error bars for deformed and undeformed observable\n $\\varepsilon_{{og}} / \\varepsilon_{{def}}$ = {err_re_og / err_re:.2f}")
    plt.legend()
    plt.show();

def save_plots(n,observable,observable_var,undeformed_obs,deformed_obs,a0,af,anorm,losses_train,losses_val,loss_name,deformation_type,title=None,path=None):
    # SETUP
    # ts = datetime.datetime.today().strftime('%Y.%m.%d_%H:%M')
    # path = os.path.join("./plots/",ts + "/")
    # os.mkdir(path)

    # VARIANCE PLOT
    epochs = len(observable)

    Nboot = 1_000
    mean_re_og, err_re_og = al.bootstrap(grab(undeformed_obs),Nboot=Nboot,f=al.rmean)
    mean_re, err_re = al.bootstrap(grab(deformed_obs),Nboot=Nboot,f=al.rmean)
    # LINEAR DEFORMATION
    if deformation_type == "Linear":
        aZ = torch.tensor(af[0]).cfloat()
        aW = torch.tensor(af[1]).cfloat()
    elif deformation_type == "Homogeneous":
        # HOMOGENEOUS DEFORMATION
        su_n = LieSU(n+1)
        aZ = su_n.embed(torch.tensor(af[0]))
        aW = su_n.embed(torch.tensor(af[1]))
    elif deformation_type == "Lattice":
        su_n = LieSU(n+1)
        # a0norms_ = grab(torch.linalg.norm(torch.tensor(a0),dim=-1))
        afnorms_ = grab(torch.linalg.norm(torch.tensor(af),dim=-1))
        a_max = af[np.unravel_index(afnorms_.argmax(),afnorms_.shape)] # def param with largest norm
        gamma = su_n.embed(torch.tensor(a_max))
        # aZ = su_n.embed(torch.tensor(af[0,0]))
        # aW = su_n.embed(torch.tensor(af[0,1]))


    # OBSERVABLE
    fig, ax = plt.subplots(nrows=2,ncols=2)

    ax[0,1].plot(anorm)
    ax[0,1].set_xlabel("epochs")
    ax[0,1].set_title(r"$\Vert a \Vert$")

    ax[0,0].plot(*np.transpose(losses_train),label="training")
    ax[0,0].plot(*np.transpose(losses_val),label="validation")
    ax[0,0].set_xlabel("epochs")
    ax[0,0].set_title(loss_name)
    ax[0,0].legend()

    ax[1,0].plot(*np.transpose([(e,z.real) for e,z in observable]),label='def')
    ax[1,0].axhline(y=mean_re_og,xmin=0,xmax=epochs,label='og',color='red')
    ax[1,0].fill_between([-100,epochs], [mean_re_og-err_re]*2, [mean_re_og+err_re]*2, alpha=0.5, color='red')
    ax[1,0].set_xlabel("epochs")
    ax[1,0].set_title(r"$\langle$" + title.split(' ')[0][:-1] + r"$\rangle$")
    ax[1,0].legend()

    ax[1,1].plot(*np.transpose([(e,z.real) for e,z in observable_var]),label='def')
    ax[1,1].axhline(y=grab(undeformed_obs.var()),xmin=0,xmax=epochs,label='og',color='red')
    ax[1,1].set_xlabel("epochs")
    ax[1,1].set_title("variance")
    ax[1,1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(path + "loss.pdf")

    # EXAMPLE LEARNED DEFORMATION PARAMETER
    if deformation_type == "Lattice":
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5))

        sns.heatmap(data=grab(gamma.real),ax=ax[0],cmap='inferno')
        ax[0].set_title(r"${\rm Re}(a_{max})$")
        sns.heatmap(data=grab(gamma.imag),ax=ax[1],cmap='inferno')
        ax[1].set_title(r"${\rm Im}(a_{max})$")

        plt.suptitle(title + ", deformation parameter")
        plt.tight_layout()

        fig.savefig(path + "deformation_params.pdf")

    elif deformation_type == "Homogeneous":
        fig, ax = plt.subplots(nrows=2,ncols=2)
    
        sns.heatmap(data=grab(aZ.real),ax=ax[0,0],cmap='coolwarm')
        ax[0,0].set_title(r"${\rm Re}(a(z))$")
        sns.heatmap(data=grab(aZ.imag),ax=ax[0,1],cmap='coolwarm')
        ax[0,1].set_title(r"${\rm Im}(a(z))$")
    
        sns.heatmap(data=grab(aW.real),ax=ax[1,0],cmap='coolwarm')
        ax[1,0].set_title(r"${\rm Re}(a(w))$")
        sns.heatmap(data=grab(aW.imag),ax=ax[1,1],cmap='coolwarm')
        ax[1,1].set_title(r"${\rm Im}(a(w))$")

        plt.suptitle(title + ", deformation parameters")
        plt.tight_layout()

        fig.savefig(path + "deformation_params.pdf");

    # ERRORBARS
    fig = plt.figure()
    plt.errorbar([0],[mean_re_og],[err_re_og],color='blue',label=r"$\varepsilon_{og}$",marker='o',capsize=2)
    plt.errorbar([1],[mean_re],[err_re],color='red',label=r"$\varepsilon_{def}$",marker='o',capsize=2)
    plt.xlim(-1,2)
    plt.xticks([],[])
    plt.title(f"error bars for deformed and undeformed observable\n $\\varepsilon_{{og}} / \\varepsilon_{{def}}$ = {err_re_og / err_re:.2f}")
    plt.legend()
    fig.savefig(path + "errorbars_comp.pdf");

    # LATTICE OF DEFORMATION PARAMETERS
    if deformation_type == "Lattice":
        fig, ax  = plt.subplots(figsize=(20,10))

        scatter_heatmap(ax,afnorms_,"after training") # color map after training
        # scatter_heatmap(ax[0],a0norms_,"before training",color_map=color_map) # use same color map

        ax.set_xlabel("Lx")
        ax.set_ylabel("Ly")
        ax.set_title(title + r", $\Vert a(x,y) \Vert$", fontsize=32,y=1.03,x=0.47)
        plt.tight_layout()

        fig.savefig(path + "deformation_params_norms.pdf", bbox_inches='tight')

def scatter_heatmap(ax, anorm, title, color_map = None, return_color_map = False, plot_cbar = True):
    x, y = np.meshgrid(np.arange(anorm.shape[1]), np.arange(anorm.shape[0]))
    x = x.flatten()
    y = y.flatten()
    colors = anorm.flatten()
    cmap = cm.inferno  
    color_map_ = color_map if color_map is not None else mcolors.Normalize(vmin=colors.min(), vmax=colors.max()) 
    mapped_colors = [cmap(color_map_(value)) for value in colors]  

    sns.scatterplot(x=x,y=y,color=mapped_colors,s=70,ax=ax,legend=False)

    ax.invert_yaxis() # match heatmap orientation
    ax.set_xticks(np.arange(anorm.shape[1]))
    ax.set_yticks(np.arange(anorm.shape[0]))
    ax.set_xticklabels(range(0,len(anorm)),rotation=-90)
    ax.set_yticklabels(range(0,len(anorm)))
    ax.margins(y=0.01,x = 0.01)
    ax.set_title(title,fontsize=24)

    sm = cm.ScalarMappable(cmap=cmap, norm=color_map_)  # color scale reference
    sm.set_array([]) 

    if plot_cbar:
        plt.colorbar(sm, ax=ax) 

    if return_color_map:
        return color_map_ 

def save_tensor(t,path,filename):
    torch.save(t, path + filename)
