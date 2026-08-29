# -*- coding: utf-8 -*-
import numpy as np

from pywarpx import particle_containers, picmi,callbacks,libwarpx
from wxutils.features.helpers import set_species_params
import wxutils.mpitools as mpit
from pywarpx.LoadThirdParty import load_cupy
from pathlib import Path
from .diag import Diagnostic1D

saveloc = Path('./diags/hist')

class CurrentAbs(Diagnostic1D):
    def __init__(self,name,species,boundary,io,save_period,**kw):
        super().__init__(name,io,save_period,**kw)
        if not isinstance(species,(list,tuple,picmi.Species)):
            raise TypeError("paritcles must be a picmi.Species or list-like of picmi.Species")
        elif isinstance(species,(picmi.Species)):
            self.speciesList = [species]

        if not isinstance(boundary,(list,tuple,str)):
            raise TypeError("boundary must be of type string, list, or tuple")
        elif not isinstance(boundary,(list,tuple,)):
            self.boundaryList = [boundary]
        else:
            self.boundaryList = boundary
        self.reduce_op = mpit.mpi.SUM
        self.speciesList = [set_species_params(_species,self.boundaryList) for _species in self.speciesList]
        self.dump_at_step_zero = False
        self.lev = 0
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        
    def post_initialize(self):
        super().post_initialize()
        self.data = self.xp.zeros((2, self.data_buffer_length), dtype=self.xp.float64)
        # self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()

        if self.save_period is None:
            # check for diagnostics and grab that value
            if len(self.sim.diagnostics)>0: 
                self.save_period = self.sim.diagnostics[0].period

    def get(self):
        '''must return tuple of (,time,value), this will get logged'''
        q = 0
        for species in self.speciesList:
            w=0
            for boundary in self.boundaryList:
                tiles = self.concat(self.buffer.get_particle_scraped_this_step(species.name, boundary, "w", self.lev))
                for tile in tiles:
                    w += tile.sum()
            q += w * species.charge
        
        t = self.sim.extension.warpx.gett_new(self.lev)
        dt = self.sim.extension.warpx.getdt(self.lev)
        return t,q/dt
    
    def log(self):
        x0,x1 = self.get()

        self.data[0, self.log_step] = x0
        self.data[1, self.log_step] = x1
        super().log()
    
    def save(self):
        """Collective flush—reduces entire buffered vector across ranks at once."""
        if self.log_step == 0:
            return
    
        # 1. Pull active local slice to CPU host memory
        active_data = self.data[:,:self.log_step]
        if hasattr(active_data, "get"):
            active_data = active_data.get()
        
        # Rows are inherently C-contiguous in memory
        x0_local = active_data[0]
        x1_local = active_data[1]
        
        # 2. Perform chunked vector reduction across MPI ranks
        if self.mpii.enabled and self.reduce_op is not None:
            x1_global = np.empty_like(x1_local) if self.mpii.is_root else None
            self.mpii.comm.Reduce(x1_local, x1_global, op=self.reduce_op, root=0)
        else:
            x1_global = x1_local
        self.io.save(self.name,(x0_local,x1_global))
        self.log_step = 0
        
        