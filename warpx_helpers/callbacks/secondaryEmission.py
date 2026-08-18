# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi
from pywarpx.LoadThirdParty import load_cupy

from warpx_helpers.methods import energy_to_velocity

constants = picmi.constants

sigma0 = 2.0
Emax0 = 300
Eth = 5.0
k1=0.62
k2=0.25 
engSecAvg = 5.0
uSecMag = energy_to_velocity(engSecAvg)
def concat(list_of_arrays):
    xp, _ = load_cupy()
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return xp.empty(0)
    else:
        return xp.concatenate(list_of_arrays)

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
def print_angles(sim,species='electrons',boundary='eb'):
    xp, _ = load_cupy()
    buffer = particle_containers.ParticleBoundaryBufferWrapper()
    lev = 0
    # xTiles = buffer.get_particle_scraped_this_step(species, boundary, "x", lev)
    # zTiles = buffer.get_particle_scraped_this_step(species, boundary, "z", lev)
    nxTiles = buffer.get_particle_scraped_this_step(species, boundary, "nx", lev)
    nzTiles = buffer.get_particle_scraped_this_step(species, boundary, "nz", lev)
    
    total_sum = 0.0
    total_count = 0
    for nxTile,nzTile in zip(nxTiles,nzTiles):
        total_sum += float(xp.arctan2(nzTile, nxTile).sum())
        total_count += len(nxTile)
    angle = total_sum / total_count if total_count > 0 else 0.0
    
    # nxTiles = concat(nxTiles)
    # nzTiles = concat(nzTiles)
    # # nx = nxTiles.mean()
    # # nz = nzTiles.mean()
    # angle = xp.arctan2(nzTiles, nxTiles)
    
    angle = xp.rad2deg(angle)
    print(angle)
    
    # running_angle.add(angle)
    # print(running_angle.mean)
    
def gen_secondary(sim,species='electrons',boundary='eb',mask=None):
    xp, _ = load_cupy()
    electrons = sim.particles.get(species)
    # pc = particle_containers.ParticleContainerWrapper(species)
    mass = constants.m_e
    dt = sim.time_step_size
    
    buffer = particle_containers.ParticleBoundaryBufferWrapper()
    lev = 0
    xTiles = buffer.get_particle_scraped_this_step(species, boundary, "x", lev)
    zTiles = buffer.get_particle_scraped_this_step(species, boundary, "z", lev)
    #yTiles = buffer.get_particle_scraped_this_step(species, boundary, "y", lev)
    uxTiles = buffer.get_particle_scraped_this_step(species, boundary, "ux", lev)
    uzTiles = buffer.get_particle_scraped_this_step(species, boundary, "uz", lev)
    nxTiles = buffer.get_particle_scraped_this_step(species, boundary, "nx", lev)
    nzTiles = buffer.get_particle_scraped_this_step(species, boundary, "nz", lev)
    wTiles = buffer.get_particle_scraped_this_step(species, boundary, "w", lev)
    dtTiles = buffer.get_particle_scraped_this_step(species, boundary, "deltaTimeScraped", lev)
    

    for xTile,zTile,uxTile,uzTile,nxTile,nzTile,wTile,dtTile in zip(xTiles,zTiles,uxTiles,uzTiles,nxTiles,nzTiles,wTiles,dtTiles):

        Ivalid = mask(xTile, zTile) if mask is not None else True
        
        u_sq = (uxTile**2 + uzTile**2)
        
        # relativistic
        u_norm_sq = u_sq / (constants.c**2)  # Dimensionless (gamma^2 * beta^2)
        gamma_minus_1 = u_norm_sq / (xp.sqrt(1.0 + u_norm_sq) + 1.0)
        eng = gamma_minus_1 * mass * (constants.c**2) / constants.q_e
        
        # non-reletivistic
        # eng = 0.5 * mass * u_sq/constants.q_e
    
        nSec = sigma0*numerical(eng,Emax0,Eth)
        I = (nSec>0.0) & Ivalid
        wSec = wTile[I]*nSec[I]
        uxSec = uSecMag * nxTile[I]
        uzSec = uSecMag * nzTile[I]
        tr = dt - dtTile[I]
        
        electrons.add_particles(
            x=xTile[I] + tr * uxSec,
            z=zTile[I] + tr * uzSec,
            ux=uxSec,
            uz=uzSec,
            w=wSec,
        )  

def initialize_surface_rho(sim,name="rho_surface"):
    rho = sim.fields.get('rho')
    sim.fields.alloc_init(name=name,
                        dir=0,
                        level=0,
                        ba=rho.box_array(),
                        dm=rho.dm(),
                        ncomp=rho.n_comp,
                        ngrow=rho.n_grow_vect,
                        initial_value=0.,
                        redistribute=True,
                        redistribute_on_remake=True,
                        checkpoint_restart=False)
    

def deposit_surface_charge(sim,name="rho_surface"):
    rho_surface = sim.fields.get(name)
    
    

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
    
