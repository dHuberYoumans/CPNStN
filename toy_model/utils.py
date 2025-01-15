import numpy as np

def grab(x): 
    return x.detach().cpu().numpy()

def safe_grab(x): 
    array = x.detach().cpu().numpy()

    if np.any(np.isnan(array)):
        print("Warning: NaN!")

    if np.any(np.isinf(array)):
        print("Warning: Inf detected in tensor!")
    return array
