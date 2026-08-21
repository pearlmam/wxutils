# -*- coding: utf-8 -*-
# import copy
# import time

import os
from mpi4py import MPI as mpi


import numpy as np
from pywarpx import particle_containers, picmi,callbacks,libwarpx
from pywarpx.LoadThirdParty import load_cupy

from wxutils.physics import energy_to_velocity
from wxutils.callbacks.utils import set_species_params
constants = picmi.constants

class RunningMean:
    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def add(self, value: float) -> float:
        if value != 0.0:
            self.count += 1
            self.mean += (value - self.mean) / self.count
        return self.mean
        
running_angle = RunningMean()
class SecondaryEmission:
    def __init__(self,species0,sigmaMax,Emax,species1=None,Emin=5.0,Eemit=5.0,mask=None,boundary="eb",dumprate=None,wThresh=1e-5):
        
        self.species0 = set_species_params(species0,boundary)
        if species1 is None:
            self.species1 = self.species0
        else:
            self.species1 = set_species_params(species1)
        
        self.sigmaMax = sigmaMax
        self.Emax = Emax
        self.Emin = Emin
        self.Eemit = Eemit
        self.Uemit = energy_to_velocity(self.Eemit)
        self.mask = mask
        self.boundary = boundary
        self.dumprate = dumprate
        self.saveloc = "./diags/fields/"
        self.wThresh = wThresh
        self.rank = mpi.COMM_WORLD.Get_rank()
        
    def pre_initialize(self,sim,rhoSurfFieldName="rho_surf"):
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installafterstep(self.gen_secondary)
        callbacks.installcallback("afterdeposition",self.deposit_surface_charge)
        
        self.sim = sim
        self.rhoSurfFieldName = rhoSurfFieldName
        self.surface_species0 = picmi.Species(name="surface_species0",
                                        mass = self.species0.mass,
                                        charge = self.species0.charge,
                                        warpx_save_particles_at_eb=False,
                                        )
        self.surface_species1 = picmi.Species(name="surface_species1",
                                        mass = self.species1.mass,
                                        charge = -self.species1.charge,
                                        warpx_save_particles_at_eb=False,
                                        )
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=[1,1], grid=self.sim.solver.grid)
        sim.add_species(
            self.surface_species0,
            layout=layout
        )
        
        sim.add_species(
            self.surface_species1,
            layout=layout
        )
    
    
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self.initialize_surface_rho(self.rhoSurfFieldName)
        self.rho_surf = self.sim.fields.get(self.rhoSurfFieldName,level=0)
        self.rho = self.sim.fields.get("rho_fp",level=0)
        self.species1_pc = self.sim.particles.get(self.species1.name)
        self.surface_species0_pc = self.sim.particles.get(self.surface_species0.name)
        self.surface_species1_pc = self.sim.particles.get(self.surface_species1.name)
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()
        
        if self.dumprate is not None:
            os.makedirs(self.saveloc,exist_ok=True)
            

    def _gen_secondary(self,x,z,y,ux,uz,uy,nx,nz,ny,w,delta_t):
        # species1_pc = self.sim.particles.get(self.species1.name)
        eng = self.get_impact_energy(ux,uz)
        nSec = self.sigmaMax*numerical(eng,self.Emax,self.Emin)
        wSec = w*nSec
        I = (wSec>self.wThresh)
        wSec = wSec[I]
        uxSec = self.Uemit * nx[I]
        uzSec = self.Uemit * nz[I]
        tr = self.sim.time_step_size - delta_t[I]
        
        self.species1_pc.add_particles(
            x=x[I] + tr * uxSec,
            z=z[I] + tr * uzSec,
            #y=y[I],
            ux=uxSec,
            uz=uzSec,
            w=wSec,
            unique_particles=True
        )  
        
        # Determine charge
        self.surface_species0_pc.add_particles(x=x,z=z,w=w,)  
        self.surface_species0_pc.deposit_charge(self.rho_surf,lev=0)
        self.surface_species0_pc.clear_particles()

        self.surface_species1_pc.add_particles(x=x[I],z=z[I],w=wSec)
        self.surface_species1_pc.deposit_charge(self.rho_surf,lev=0)
        self.surface_species1_pc.clear_particles()
    
    
    def gen_secondary(self):
        name = self.species0.name
        lev = 0
        x = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "x", lev))
        z = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "z", lev))
        # y = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "y", lev))
        ux = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "ux", lev))
        uz = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "uz", lev))
        nx = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "nx", lev))
        nz = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "nz", lev))
        w = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "w", lev))
        delta_t = self.concat(self.buffer.get_particle_scraped_this_step(name, self.boundary, "deltaTimeScraped", lev))
    
        I = self.mask(x, z) if self.mask is not None else True
        self._gen_secondary(x[I], z[I], None, ux[I], uz[I], None, nx[I], nz[I], None, w[I], delta_t[I])
        
    def get_impact_energy(self,ux,uz,uy=None,relativistic=True):
        u_sq = (ux**2 + uz**2)
        mass = self.species0.mass
        if relativistic == True:
            u_norm_sq = u_sq / (constants.c**2)  # Dimensionless (gamma^2 * beta^2)
            gamma_minus_1 = u_norm_sq / (self.xp.sqrt(1.0 + u_norm_sq) + 1.0)
            eng = gamma_minus_1 * mass * (constants.c**2) / constants.q_e
        else:
            eng = 0.5 * mass * u_sq/constants.q_e
        return eng
    
    
    def initialize_surface_rho(self,name="rho_surf"):
        rho = self.sim.fields.get('rho_fp',level=0)
        self.sim.fields.alloc_init(name=name,
                            level=0,
                            ba=rho.box_array(),
                            dm=rho.dm(),
                            ncomp=rho.n_comp,
                            ngrow=rho.n_grow_vect,
                            initial_value=0.,
                            redistribute=True,
                            redistribute_on_remake=True,
                            checkpoint_restart=False)

    def deposit_surface_charge(self):
        self.rho.saxpy(1.0, self.rho_surf, 0, 0, 1, 0)
        #self.save_field(self.rho,"rho_total")
    
    def save_field(self,field,savename):
        if self.dumprate is not None:
            i = self.sim.extension.warpx.getistep(0)
            if i % self.dumprate == 0:
                field_data = field[...]
                if hasattr(field_data, "get"):
                    field_data = field_data.get()
                if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
                    np.save(f"{self.saveloc}/{savename}_{i}.npy", field_data)
    
    def concat(self, list_of_arrays,*args,**kwargs):
        if len(list_of_arrays) == 0:
            # Return a 1d array of size 0
            return self.xp.empty(0)
        else:
            return self.xp.concatenate(list_of_arrays,*args,**kwargs)
    
    




###### secondary emission models
k1=0.62
k2=0.25 
def numerical(x,Emax0,Eth):
    # Detect backend dynamically (NumPy or CuPy)
    xp, _ = load_cupy()
    
    # Initialize output array matching input shape and type
    out = xp.zeros_like(x, dtype=xp.float64)
    x = (x - Eth) / (Emax0 - Eth)
    # Define boolean condition masks
    m1 = (x > 0) & (x <= 1.0)
    m2 = (x > 1.0) & (x <= 3.6)
    m3 = x > 3.6
    
    # Slice subsets once to avoid domain errors/warnings (e.g., negative powers)
    x1, x2, x3 = x[m1], x[m2], x[m3]
    
    # Vectorized evaluation only on matching slices
    out[m1] = (x1 * xp.exp(1.0 - x1))**0.56
    out[m2] = (x2 * xp.exp(1.0 - x2))**0.25
    out[m3] = 1.125 / (x3**0.35)
    
    return out       
    
def lyeDekker(engInc,EMax):
    xp, _ = load_cupy()
    return 1.379*(1-xp.exp(-(1.844*engInc/EMax)**1.35))/(1.844*engInc/EMax)**0.35

def vaughan(engInc,EMax,E0,k1,k2):
    xp, _ = load_cupy()
    v = xp.maximum(0.0,(engInc-E0)/(EMax-E0))
    k = (k1+k2)/2-(k1-k2)/xp.pi*xp.arctan(xp.pi*xp.log((engInc-E0)/(EMax-E0)))
    return (v*xp.exp(1-v))**k
    
    
def vaughanCombined(engInc,EMax,E0,k1,k2):
    xp, _ = load_cupy()
    delta2 = lyeDekker(engInc,EMax)
    delta1 = vaughan(engInc,EMax,E0,k1,k2)
    v = xp.maximum(0.0,(engInc-E0)/(EMax-E0))
    v3 = (1/k2-0.25)
    return (delta1+delta2)/2 - (delta1-delta2)/xp.pi*xp.arctan(xp.pi*xp.log(v/v3))
    
