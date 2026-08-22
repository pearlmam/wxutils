# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi,callbacks,libwarpx

class SurfaceDeposit():
    def __init__(self,rhoSurf="rho_surf",species=None):
        self.rhoSurf = rhoSurf
        if not isinstance(species,(list,tuple,picmi.Species)):
            raise TypeError("species must be a picmi.Species or list-like of picmi.Species")
        elif isinstance(species,(picmi.Species)):
            self.speciesList = [species]
        
        
    def pre_initialize(self,sim):
        self.sim = sim
        self.sim.extensions.warpx