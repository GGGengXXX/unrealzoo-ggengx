import numpy as np

def getP(rate):
    if np.random.rand() < rate:
        return True
    else:
        return False