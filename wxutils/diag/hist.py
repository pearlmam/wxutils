# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi,callbacks,libwarpx
from wxutils.callbacks.utils import set_species_params
from pywarpx.LoadThirdParty import load_cupy



class CurrentAbs():
    def __init__(self,species,boundary):
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
    def initialize(self,sim):
        self.xp,_ = load_cupy()
        self.sim = sim
        self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()
        callbacks.installafterstep(self.get)
        
    def get(self):
        lev=0
        q = 0
        for species in self.speciesList:
            w=0
            for boundary in self.boundaryList:
                tiles = self.concat(self.buffer.get_particle_scraped_this_step(species.name, boundary, "w", lev))
                for tile in tiles:
                    w += tile.sum()
            q += w * species.charge
        print(q)
        #return q
        
        
    def concat(self, list_of_arrays,*args,**kwargs):
        if len(list_of_arrays) == 0:
            # Return a 1d array of size 0
            return self.xp.empty(0)
        else:
            return self.xp.concatenate(list_of_arrays,*args,**kwargs)
        
        