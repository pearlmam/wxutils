# -*- coding: utf-8 -*-

import numpy as np
from pywarpx.LoadThirdParty import load_cupy

def to_cpu_array(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def concat(list_of_arrays):
    xp,_ = load_cupy
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return xp.empty(0)
    else:
        return xp.concatenate(list_of_arrays)
        