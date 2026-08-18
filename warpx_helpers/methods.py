# -*- coding: utf-8 -*-

import numpy as np
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

def energy_to_velocity(energy_ev,m0=9.1093837139e-31):
    """
    Converts beam kinetic energy to relativistic velocity.
    
    Parameters:
    energy_ev (float): Kinetic energy of the beam in electronvolts (eV).
    
    Returns:
    float: Velocity of the electron in meters per second (m/s).
    """

    # Calculate electron rest mass energy in Joules
    E_rest_joules = m0 * (constants.c ** 2)
    
    # Convert input kinetic energy from eV to Joules
    E_k_joules = energy_ev * constants.q_e
    
    # Calculate velocity using the relativistic formula
    # v = c * sqrt(1 - (E_rest / (E_k + E_rest))^2)
    gamma_inv = E_rest_joules / (E_k_joules + E_rest_joules)
    velocity = constants.c * np.sqrt(1.0 - (gamma_inv ** 2))
    
    return velocity

def to_cpu_array(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)



        
