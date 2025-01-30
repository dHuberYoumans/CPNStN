import torch

def loss(obs):
    return obs.var()

def rloss(obs):
    assert obs.is_complex(), "expect complex input"
    return obs.real.var()

def iloss(obs):
    assert obs.is_complex, "expect complex input"
    return obs.imag.var()

def logloss(obs):
    return torch.log(obs.var())

def rlogloss(obs):
    return torch.log(obs.real.var())

def ilogloss(obs):
    return torch.log(obs.imag.var())