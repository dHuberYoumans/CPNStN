import numpy as np
import time
import torch 

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
