# -*- coding: utf-8 -*-

from pywarpx import picmi,geometry,amr
constants = picmi.constants

def get_grid_cell_sizes():
    """
    Returns a list of cell sizes [dx, dy, dz] based on the 
    current simulation domain and number of cells.
    """
    lo = geometry.prob_lo
    hi = geometry.prob_hi
    n_cell = amr.n_cell
    
    # Calculate spacing: (Domain Length) / (Number of Cells)
    # This works automatically for 1D, 2D, or 3D
    cell_sizes = [(hi[i] - lo[i]) / n_cell[i] for i in range(len(n_cell))]
    
    return cell_sizes

