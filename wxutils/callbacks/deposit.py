# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi,callbacks
from pywarpx.LoadThirdParty import load_cupy
from wxutils.core import CallbackBase

class Deposit(CallbackBase):
    def __init__(self,**kw):
        
        self.rho = kw.pop("rho","rho_fp")
        self.method = kw.pop("method","areaWeighting")
        self.speciesList = kw.pop("species",None)
        if not isinstance(self.speciesList,(list,tuple,picmi.Species)):
            raise TypeError("species must be a picmi.Species or list-like of picmi.Species")
        elif isinstance(self.speciesList,(picmi.Species)):
            self.speciesList = [self.speciesList]
        
        if self.method == "surfaceWeighting":
            self.sinkList = kw.pop("sink",None)
            if not isinstance(self.sinkList,(list,tuple,picmi.ParticleSink,None)):
                raise TypeError("sink must be a picmi.ParticleSink or list-like of picmi.ParticleSink")
            elif isinstance(self.sinkList,(picmi.ParticleSink)):
                self.sinkList = [self.sinkList]
        
        self.sim = kw.pop("sim",None)
        if self.sim is not None:
            self.pre_initialize(self.sim)
        
        self.lev = 0

        
    def pre_initialize(self,sim):
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        self.sim = sim
        if (self.speciesList is None):
            if (self.sim.species is not None):
                self.speciesList = self.sim.species
            else:
                raise TypeError("There are no species to deposit")
        
    def post_initialize(self):
        self.xp,_ = load_cupy()
        self.grid_info()   # CallbackBase method: adds grid info to self
        self.rho = self.sim.fields.get(self.rho,level=self.lev)
        self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        self._bind_method()
        # callbacks.installcallback("afterdeposition", self.deposit_charge)
        print("Deposit: post_initialized")
    
    def _bind_method(self):
        if self.method == "areaWeighting":
            self.deposit_charge = self.area_weighting
        elif self.method == "surfaceWeighting":
            self.deposit_charge = self.surface_weighting
            self.sdfList = []
            for sink in self.sinkList: 
                fieldName = "distance_to_" + sink.name
                self.sdfList.append(self.sim.fields.get(fieldName,level=self.lev))
        else:
            self.deposit_charge = self.warpx_deposit_charge
        
    def get_pc_data(self, species_pc, variables, lev=0):
        
        return tuple(
            self.concat(species_pc.get_particle_scraped_this_step(var, lev))
            for var in variables
        )
    
    def area_weighting(self,x, y, w, q, rho, xmin, ymin, dx, dy):
        """
        Deposits charge onto a 2D numpy grid 'rho' [nx, ny].
        """
        cell_area = dx * dy
        q_eff = (q * w) / cell_area
    
        # Fractional coordinates
        x_cell = (x - xmin) / dx
        y_cell = (y - ymin) / dy
    
        i = self.xp.floor(x_cell).astype(int)
        j = self.xp.floor(y_cell).astype(int)
    
        fx = x_cell - i
        fy = y_cell - j
    
        # Bilinear/Area shape factors
        w00 = (1.0 - fx) * (1.0 - fy) * q_eff
        w10 = fx * (1.0 - fy) * q_eff
        w01 = (1.0 - fx) * fy * q_eff
        w11 = fx * fy * q_eff
    
        # Accumulate into grid (np.add.at handles duplicate index updates safely)
        self.xp.add.at(rho, (i, j), w00)
        self.xp.add.at(rho, (i + 1, j), w10)
        self.xp.add.at(rho, (i, j + 1), w01)
        self.xp.add.at(rho, (i + 1, j + 1), w11)
        
    def surface_weighting(self,
        x, y, w, q, nx, ny, rho, is_dielectric, xmin, ymin, dx, dy):
        """
        Deposits charge exclusively into dielectric nodes.
        - nx, ny: Normal unit vectors pointing INTO the dielectric.
        - is_dielectric: 2D boolean mask (True = Dielectric, False = Vacuum).
        """
        cell_area = dx * dy
        q_eff = (q * w) / cell_area
    
        # Nudge scraped particle positions slightly into dielectric along surface normal
        nudge = 0.5 * min(dx, dy)
        x_pos = x + nudge * nx
        y_pos = y + nudge * ny
    
        x_cell = (x_pos - xmin) / dx
        y_cell = (y_pos - ymin) / dy
    
        i = self.xp.floor(x_cell).astype(int)
        j = self.xp.floor(y_cell).astype(int)
    
        fx = x_cell - i
        fy = y_cell - j

        # Standard bilinear shape factors
        s00 = (1.0 - fx) * (1.0 - fy)
        s10 = fx * (1.0 - fy)
        s01 = (1.0 - fx) * fy
        s11 = fx * fy
    
        # Dielectric mask at corner nodes
        m00 = is_dielectric[i, j].astype(float)
        m10 = is_dielectric[i + 1, j].astype(float)
        m01 = is_dielectric[i, j + 1].astype(float)
        m11 = is_dielectric[i + 1, j + 1].astype(float)
    
        # Apply mask and compute normalization denominator
        w00 = s00 * m00
        w10 = s10 * m10
        w01 = s01 * m01
        w11 = s11 * m11
    
        w_tot = w00 + w10 + w01 + w11
    
        # Safe division: zero out weights where no dielectric node is in stencil
        valid = w_tot > 0
        w_tot_safe = self.xp.where(valid, w_tot, 1.0)
    
        w00 = self.xp.where(valid, (w00 / w_tot_safe) * q_eff, 0.0)
        w10 = self.xp.where(valid, (w10 / w_tot_safe) * q_eff, 0.0)
        w01 = self.xp.where(valid, (w01 / w_tot_safe) * q_eff, 0.0)
        w11 = self.xp.where(valid, (w11 / w_tot_safe) * q_eff, 0.0)
    
        # Deposit into dielectric nodes
        self.xp.add.at(rho, (i, j), w00)
        self.xp.add.at(rho, (i + 1, j), w10)
        self.xp.add.at(rho, (i, j + 1), w01)
        self.xp.add.at(rho, (i + 1, j + 1), w11)
        
    def warpx_deposit_charge(self,):
        for species_pc in self.species_pc:
            species_pc.deposit_charge(self.rho,lev=self.lev)