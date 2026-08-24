# -*- coding: utf-8 -*-

import shutil

from pywarpx.LoadThirdParty import load_cupy
from pywarpx import picmi
constants = picmi.constants
import wxutils.mpitools as mpit

def concat(list_of_arrays):
    xp,_ = load_cupy
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return xp.empty(0)
    else:
        return xp.concatenate(list_of_arrays)
    
def to_cpu(arr):
    """Converts a CuPy array to NumPy, or returns the NumPy array as-is."""
    return arr.get() if hasattr(arr, 'get') else arr

def delete_diagnostics():
    if mpit.get_rank() == 0:
        shutil.rmtree('./diags',ignore_errors=True)

