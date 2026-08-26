# -*- coding: utf-8 -*-

import shutil

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