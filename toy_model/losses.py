import torch

def loss(obs):
    return obs.var()

def loss_re(obs):
    assert obs.is_complex(), "expect complex input"
    return obs.real.var()

def loss_im(obs):
    assert obs.is_complex, "expect complex input"
    return obs.imag.var()

def logloss(obs):
    return torch.log(obs.var())