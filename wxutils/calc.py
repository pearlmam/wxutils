# -*- coding: utf-8 -*-
from pywarpx import picmi
constants = picmi.constants()

from scipy import constants

def flux_from_current(I, area, maskFactor=1.0, q=None, fudgeFactor=2.0):
    """
    Calculates physical particle flux [m^-2 s^-1] for WarpX AnalyticFluxDistribution.

    Parameters
    ----------
    I : float
        Total current in Amps (C/s).
    area : float
        Emission area in m^2.
    maskFactor : float, optional
        if emission is masked, need to inject more particles. 
        maskFactor = totalSurfaceArea/areaEmit 
    q : float, optional
        Particle charge magnitude in Coulombs. Defaults to elementary charge e.
    fudgeFactor : float, optional
        For some reason, only half the particles are emitted.
        I need to multiple by 2 to correct flux and n_macroparticles_per_cell should be doubled 
    
    Returns
    -------
    flux_val : float
        Numerical value of particle flux.
    flux_str : str
        Formatted string expression for WarpX.
    """
    if q is None:
        q = constants.e
    else:
        q = abs(q)
        
    # Physical particle flux: [particles / (s * m^2)]
    flux_val = I / (q * area)*maskFactor*fudgeFactor
    flux_str = f"{flux_val:.16e}"
    
    return flux_val
    
    
    