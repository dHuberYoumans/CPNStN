import numpy as np
import time

def grab(x): 
    return x.detach().cpu().numpy()

def safe_grab(x): 
    array = x.detach().cpu().numpy()

    if np.any(np.isnan(array)):
        print("Warning: NaN!")

    if np.any(np.isinf(array)):
        print("Warning: Inf detected in tensor!")
    return array

import time

class Timer:
    """
    A simple context manager for timing code execution.
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
