# -*- coding: utf-8 -*-

from pywarpx import picmi
constants = picmi.constants
from wxutils.utils import get_xp,rng_normal

def energy_to_velocity(energy_ev,mass=9.1093837139e-31,relativistic=True,xp=None):
    """
    Converts beam kinetic energy to relativistic velocity.
    
    Parameters:
    energy_ev (float): Kinetic energy of the beam in electronvolts (eV).
    
    Returns:
    float: Velocity of the electron in meters per second (m/s).
    """
    xp = get_xp(xp)
            
    # Convert input kinetic energy from eV to Joules
    eng_J = energy_ev * constants.q_e
    
    if relativistic:
        # Calculate electron rest mass energy in Joules
        eng_rest_J = mass * (constants.c ** 2)
    
        # Calculate velocity using the relativistic formula
        # v = c * sqrt(1 - (E_rest / (E_k + E_rest))^2)
        
        ## first way
        # gamma_inv = eng_rest_J / (eng_J + eng_rest_J)
        # velocity = constants.c * xp.sqrt(1.0 - (gamma_inv ** 2))
        
        # Numerically stable form of (1 - 1/gamma^2) ???
        beta_sq = (eng_J * (eng_J + 2 * eng_rest_J)) / ((eng_J + eng_rest_J) ** 2)
        velocity = constants.c * xp.sqrt(beta_sq)
    else:
        velocity = xp.sqrt(2 * eng_J / mass)
    
    return velocity

## TODO this should use a coordinate transformation function, make one.
def apply_angular_dist(u_mag, nx, nz, sigma_theta, rng=None,xp=None):
    """
    Applies angular distribution to baseline direction vectors and returns velocity components.
    
    Parameters:
    u_mag : ndarray
        Velocity magnitudes (m/s).
    nx, ny : float or ndarray
        Baseline normal vector components (must be normalized).
    sigma_theta : float
        Standard deviation of angular spread in radians.
    """
    xp = get_xp(xp)
    if rng is None:
        delta_theta = xp.random.normal(loc=0.0, scale=sigma_theta, size=len(u_mag))
    else:
        delta_theta = rng_normal(rng,loc=0.0, scale=sigma_theta, size=len(u_mag))
    
    # 2. Compute deviation trig terms
    cos_d = xp.cos(delta_theta)
    sin_d = xp.sin(delta_theta)
    
    # 3. Rotate base normal by delta_theta (no arctan2 required)
    nx_new = nx * cos_d - nz * sin_d
    nz_new = nx * sin_d + nz * cos_d
    
    # 4. Scale by speed to get velocity components
    vx = u_mag * nx_new
    vz = u_mag * nz_new
    
    return vx, vz



    