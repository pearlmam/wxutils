# -*- coding: utf-8 -*-

from pywarpx import particle_containers, picmi,callbacks
from pywarpx.LoadThirdParty import load_cupy
from wxutils.core import CallbackBase
from wxutils.utils import mpiprint

class Deposit(CallbackBase):
    def __init__(self,**kw):
        
        self.rhoName = kw.pop("rho","rho_fp")
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
        
        self.use_rho_temp = True
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
        self.rho = self.sim.fields.get(self.rhoName,level=self.lev)
        self._rho = self._initialize_rho_temp()
        self.geom = self.sim.extension.warpx.Geom(self.lev)
        self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        # self.species_pcw = [particle_containers.ParticleContainerWrapper(species.name) for species in self.speciesList]
        self.pcVars = ["x","z","w"]
        self._bind_method()
        # callbacks.installcallback("afterdeposition", self.deposit_charge)
        
    
    def _initialize_rho_temp(self):
        '''
        Temp rho field might be needed to sum and fill boundaries.
        '''
        if self.use_rho_temp:
            self._rhoName = f"_{self.rhoName}"
            if self.sim.fields.has(self._rhoName,self.lev):
                # TODO should Exception go here
                return
            self._rho = self.sim.fields.alloc_init(name=self._rhoName,
                                level=self.lev,
                                ba=self.rho.box_array(),
                                dm=self.rho.dm(),
                                ncomp=self.rho.n_comp,
                                ngrow=self.rho.n_grow_vect,
                                initial_value=0.,
                                redistribute=True,
                                redistribute_on_remake=True,
                                checkpoint_restart=False)
        else:
            self._rho = self.rho
        return self._rho
    
    
    def _bind_method(self):
        if self.method == "areaWeighting":
            self._deposit_charge = self.area_weighting
        elif self.method == "surfaceWeighting":
            self._deposit_charge = self.surface_weighting
            self.sdfList = []
            for sink in self.sinkList: 
                fieldName = "distance_to_" + sink.name
                self.sdfList.append(self.sim.fields.get(fieldName,level=self.lev))
        else:
            self._deposit_charge = self.warpx_deposit_charge
        
    def get_pc_data(self, species_pc, variables, lev=0):
        data_arrays = species_pc.get_particle_real_arrays('x', lev)
        return tuple(self.concat(data_arrays) for var in variables)
    
    def deposit_charge(self,):
        self._deposit_charge()
        if self.use_rho_temp:
            self._rho.sum_boundary(self.geom.periodicity())
            # self._rho.fill_boundary(self.geom.periodicity()) # not needed?
            self.rho.saxpy(1.0, self._rho, 0, 0, 1, 0)
            # self.rho.sum_boundary(self.geom.periodicity())  # not needed
            # self.rho.fill_boundary(self.geom.periodicity()) # not needed
            self._rho.set_val(0.0)
    
    def warpx_deposit_charge(self,):
        #mpiprint(f"shape: {self.rho.shape}",ranks="all")
        for species_pc in self.species_pc:
            species_pc.deposit_charge(self._rho,lev=self.lev)
            
    def area_weighting(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            # ## doesnt work because Nested or multiple active MFIters is not supported by default
            # ## This can be changed by calling MFIter::allowMultipleMFIters(true) !!!
            # for pti, rho_arr in zip(species_pc.iterator(level=self.lev), self.rho.to_xp()):
            #     self._area_weighting(pti,rho_arr, q, self.lev)
            rho_arrs = self._rho.to_xp()
            for pti in species_pc.iterator(level=self.lev):
                # fab_idx = pti.index  # this returns global index! on rank 1 owns grid one but len(rho_arrs)=1
                # rho_arr = rho_arrs[fab_idx]
                rho_arr = self._rho.array(pti).to_xp()
                self._area_weighting(pti,rho_arr, q, self.lev)
            
            
    def _area_weighting(self, pti, rho_arr, q,lev=0):
        """
        Deposits charge onto a 2D numpy grid 'rho' [nx, nz].
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
        # mpiprint(f"pti index {pti.index}: min max:{min(z):.5} {max(z):.5}","all")
        dx, dz = self.dxyz
        xmin, zmin = self.lower_bound
        ng_x = self._rho.n_grow_vect[0]
        ng_z = self._rho.n_grow_vect[1]

        # Get the global cell index offset for this specific grid box
        fab_lo = pti.validbox().small_end
        fab_lo_x, fab_lo_z = fab_lo[0], fab_lo[1]
        
        #mpiprint(f"pti index {pti.index}: rho shape {rho_arr.shape}","all")
        #mpiprint(f"pti index {pti.index}: lo_z {fab_lo_z}","all")
        # mpiprint(f"pti index {pti.index}: ng_z {ng_z}","all")
        # Convert particle positions to local array indices (accounting for offset + ghost cells)
        x_cell = (x - xmin) / dx - fab_lo_x + ng_x
        z_cell = (z - zmin) / dz - fab_lo_z + ng_z

        i = self.xp.floor(x_cell).astype(int)
        j = self.xp.floor(z_cell).astype(int)

        fx, fz = x_cell - i, z_cell - j
        cell_area = dx * dz
        q_eff = (q * w) / cell_area

        w00 = (1.0 - fx) * (1.0 - fz) * q_eff
        w10 = fx * (1.0 - fz) * q_eff
        w01 = (1.0 - fx) * fz * q_eff
        w11 = fx * fz * q_eff

        # Slice 4D array (nx, nz, 1, 1) down to 2D
        fab_view = rho_arr[:, :, 0, 0] if rho_arr.ndim == 4 else rho_arr

        # Accumulate into local FAB array
        self.xp.add.at(fab_view, (i, j), w00)
        self.xp.add.at(fab_view, (i + 1, j), w10)
        self.xp.add.at(fab_view, (i, j + 1), w01)
        self.xp.add.at(fab_view, (i + 1, j + 1), w11)
        
    def _surface_weighting(self,
        x, z, w, q, nx, nz, rho, is_dielectric, xmin, zmin, dx, dz):
        """
        Deposits charge exclusively into dielectric nodes.
        - nx, ny: Normal unit vectors pointing INTO the dielectric.
        - is_dielectric: 2D boolean mask (True = Dielectric, False = Vacuum).
        """
        cell_area = dx * dz
        q_eff = (q * w) / cell_area
    
        # Nudge scraped particle positions slightly into dielectric along surface normal
        nudge = 0.5 * min(dx, dz)
        x_pos = x + nudge * nx
        z_pos = z + nudge * nz
    
        x_cell = (x_pos - xmin) / dx
        z_cell = (z_pos - zmin) / dz
    
        i = self.xp.floor(x_cell).astype(int)
        j = self.xp.floor(z_cell).astype(int)
    
        fx = x_cell - i
        fz = z_cell - j

        # Standard bilinear shape factors
        s00 = (1.0 - fx) * (1.0 - fz)
        s10 = fx * (1.0 - fz)
        s01 = (1.0 - fx) * fz
        s11 = fx * fz
    
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
        
    
        