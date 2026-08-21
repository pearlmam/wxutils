# -*- coding: utf-8 -*-

from pywarpx import particle_containers,callbacks
from pywarpx.LoadThirdParty import load_cupy

class BufferManager: 
    '''
    This is for future more complex buffer managment
    '''
    def __init__(self,sim,clear_period=1):
        if clear_period is None:
            clear_period = 0
        self.clear_period = clear_period
        self.pre_initialize(sim)
        self.lev = 0
    def pre_initialize(self,sim):
        self.sim = sim
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        if self.clear_period > 0:
            callbacks.installafterstep(self.clear)
        
    def post_initialize(self):
        # self.xp, _ = load_cupy()
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()
        
    def clear(self):
        step = self.sim.extension.warpx.getistep(self.lev)
        if step%self.clear_period == 0:
            self.buffer.clear_buffer()
            
def setup_buffer_manager(sim, clear_period=1):
    '''
    Same buffer setup as above but as a function.
    '''
    if not clear_period or clear_period <= 0:
        return

    lev = 0
    state = {"buffer": None, "xp": None}

    def post_initialize():
        state["xp"], _ = load_cupy()
        state["buffer"] = particle_containers.ParticleBoundaryBufferWrapper()

    def clear():
        step = sim.extension.warpx.getistep(lev)
        if step % clear_period == 0 and state["buffer"] is not None:
            state["buffer"].clear_buffer()

    callbacks.installcallback("beforeInitEsolve", post_initialize)
    callbacks.installafterstep(clear)
    
    
    