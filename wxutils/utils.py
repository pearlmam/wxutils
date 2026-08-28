# -*- coding: utf-8 -*-

import sys
import json

from pywarpx.LoadThirdParty import load_cupy
from pywarpx import picmi
constants = picmi.constants
import wxutils.mpitools as mpit
import numpy as np
import os
import fnmatch

def concat(list_of_arrays,xp=None):
    '''
    DO NOT USE, anything that needs this should be derived from a base class
    '''
    xp = get_xp(xp)
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return xp.empty(0)
    else:
        return xp.concatenate(list_of_arrays)
    
def to_cpu(arr):
    """Converts a CuPy array to NumPy, or returns the NumPy array as-is."""
    return arr.get() if hasattr(arr, 'get') else arr

def delete_diagnostics(root_dir='./diags', ignore_list=[]):
    if mpit.get_rank() == 0:
        rmtree_except_patterns(root_dir,ignore_list)
        
def rmtree_except_patterns(root_dir, ignore_list=[],remove_empty=True):
    # Must use topdown=True to modify dirnames in-place and skip directories
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        
        # 1. Modify dirnames in-place to ignore specific folders
        # This skips scanning or deleting the folders entirely
        dirnames[:] = [d for d in dirnames if d not in ignore_list and not any(fnmatch.fnmatch(d, p) for p in ignore_list)]
        
        # 2. Check and delete files
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            # Determine if the file matches any ignore pattern
            should_ignore = any(fnmatch.fnmatch(filename, pattern) for pattern in ignore_list)
            
            if not should_ignore:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
    if remove_empty:
        # 3. Second pass: Clean up leftover empty directories (bottom-up)
        # This prevents deleting parent folders that contain ignored subfolders
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            # Check if this directory itself is on the ignore list
            if os.path.basename(dirpath) in ignore_list:
                continue
                
            for dirname in dirnames:
                dir_to_remove = os.path.join(dirpath, dirname)
                if dirname not in ignore_list:
                    try:
                        os.rmdir(dir_to_remove)
                    except OSError:
                        pass # Folder is not empty because it contains kept files

def get_xp(xp=None):
    """
    Resolves array module (NumPy/CuPy).
    Order: Explicit kwarg -> Input argument module -> WarpX load_cupy() -> NumPy fallback.
    """
    if xp is not None:
        return xp
      
    # Try WarpX hardware-agnostic loader if initialized
    try:
        xp, _ = load_cupy()   # fails if WarpX is not initialized
        return xp
    except Exception:
        pass
        
    # Default fallback for CPU floats/arrays before WarpX init
    return np


MAX_LEN = 10  # Max elements before truncating lists or arrays
def sanitize(val):
    # Handle NumPy arrays
    if isinstance(val, np.ndarray):
        if val.size <= MAX_LEN:
            return val.tolist()
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"

    # Handle NumPy scalar types (e.g., np.float64, np.int32)
    if isinstance(val, (np.number, np.bool_)):
        return val.item()

    # Handle Lists & Tuples
    if isinstance(val, (list, tuple)):
        if len(val) <= MAX_LEN:
            return [sanitize(x) for x in val]
        return f"<list len={len(val)} preview={val[:3]}...>"

    # Handle Dicts
    if isinstance(val, dict):
        return {k: sanitize(v) for k, v in val.items() if isinstance(k, str)}

    # Native primitive types
    if isinstance(val, (int, float, str, bool, type(None))):
        return val

    # Ignore complex objects, functions, modules, etc.
    return None

def save_pywarpx_inputs(filename='pywarpx_used_inputs.json'):
    caller_globals = sys._getframe(1).f_globals
    state = {}
    for k, v in caller_globals.items():
        if k.startswith("_") or callable(v):
            continue
        
        clean_val = sanitize(v)
        if clean_val is not None:
            state[k] = clean_val

    with open(filename, "w") as f:
        json.dump(state, f, indent=4)

def get_warpx_axis_labels(dims):
    dims = str(dims)
    mapping = {
        "3": ["x", "y", "z"],
        "2": ["x", "z"],
        "RZ": ["r", "z"],
        "1": ["z"],
        }

    if dims in mapping:
        return mapping[dims]
    raise ValueError(f"Unsupported WarpX geometry dimension: {dims}")
    
if __name__ == "__main__":
    get_warpx_axis_labels(2)