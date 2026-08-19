# -*- coding: utf-8 -*-

from pywarpx import picmi
import np
constants = picmi.constants


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