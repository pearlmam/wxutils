# -*- coding: utf-8 -*-
from .diag import Diagnostic1D
# import wxutils.mpitools as mpit
from wxutils.utils import to_cpu
from wxutils.features.helpers import set_species_params
import numpy as np
from collections import deque

class ParticleHistory(Diagnostic1D):
    def __init__(self, particle, name, io,save_period,**kw):
        super().__init__(name, io=io,save_period=save_period,**kw)
        self.particle = set_species_params(particle)
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)

    def post_initialize(self):
        super().post_initialize()
        self.particle_data = self.sim.particles.get(self.particle.name)
        
class ParticleRate(ParticleHistory):
    def __init__(self, particle, name, io,save_period,**kw):
        super().__init__(particle,name, io=io,save_period=save_period,**kw)
        self.N = 0
    def pre_initialize(self,sim):
        super().pre_initialize(sim)

    def post_initialize(self):
        super().post_initialize()
        self.dt = self.sim.extension.warpx.getdt(0)
        
    def get(self):
        time = self.sim.extension.warpx.gett_new(0)
        self.Nold = self.N
        self.N = self.particle_data.total_number_of_particles()
        rate = (self.N-self.Nold)/self.dt
        #print(dir(self.data))
        return (time,rate)
        
