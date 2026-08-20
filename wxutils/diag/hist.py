# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi,callbacks,libwarpx
from wxutils.callbacks.utils import set_species_params
from pywarpx.LoadThirdParty import load_cupy
from wxutils.diag.store import Diagnostic1D
from pathlib import Path

saveloc = Path('./diags/hist')

class CurrentAbs():
    def __init__(self,species,boundary,name=None,interval=None,nsteps=None):
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
    
        self.speciesList = [set_species_params(_species,self.boundaryList) for _species in self.speciesList]
        self.lev = 0
        
        if name is None:
            raise TypeError("path must be defined, will develope generic naming soon")
        self.path = saveloc /Path(name)
        self.interval = interval
        self.nsteps = nsteps
        
        
    def post_initialize(self,sim):
        self.xp,_ = load_cupy()
        self.sim = sim
        self.data = self.xp.array([])
        self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()
        callbacks.installafterstep(self.get)
        if self.interval is None:
            # check for diagnostics and grab that value
            if len(self.sim.diagnostics)>0: 
                self.interval = self.sim.diagnostics[0].period
        self.store = Diagnostic1D(self.path,nsteps=self.nsteps,interval=self.interval)
        
    def get(self):
        
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
        self.store.log(q/dt,t)
        
        
    def concat(self, list_of_arrays,*args,**kwargs):
        if len(list_of_arrays) == 0:
            # Return a 1d array of size 0
            return self.xp.empty(0)
        else:
            return self.xp.concatenate(list_of_arrays,*args,**kwargs)
        
        