import numpy as np
import os


def areaCount(nparr, baseSize=10):
    """

    """
    base = np.sum(nparr == 2)
    target = np.sum(nparr == 1)
    return target / base * baseSize
