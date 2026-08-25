# -*- coding: utf-8 -*-

from pywarpx import picmi,callbacks
from pywarpx.LoadThirdParty import load_cupy
from wxutils.core import CallbackBase
import wxutils.mpitools as mpit
from wxutils.features.helpers import get_valid_region

class Deposit(CallbackBase):
    def __init__(self,**kw):
        
        self.rhoName = kw.pop("rho","rho_fp")
        self.method = kw.pop("method","areaWeighting")
        self.speciesList = kw.pop("species",None)
        self.persistent_charge = kw.pop("persistent_charge",None)
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
        self.exterior = None
        self.debug = False
        
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
        if self.persistent_charge:
            self._rhoName = f"_{self.rhoName}"
            self._rho = self._create_scaler_field(self._rhoName,self.rho)
        else:
            self._rho = self.rho
        return self._rho
    
    def _initialize_exterior_mask(self):
        '''
        This creates a field where values >0 are outside vacuum
        
        because sdf and rho fields can have different number of guards, 
        the exterior field updates at valid locations and fills the boundary after
        '''
        self.exterior = self._create_scaler_field(f"exterior_for_{self.rhoName}",self.rho) # unique name
        ng_ext = self.exterior.n_grow_vect
        for ext_arr in self.exterior.to_xp():
            ext_arr.fill(0.0)
        
        for sink in self.sinkList: 
            fieldName = "distance_to_" + sink.name
            sdf = self.sim.fields.get(fieldName,level=self.lev)
            ng_sdf = sdf.n_grow_vect
            for mfi in self.exterior:  # cant use zip, AMRex MFIter only allows 1 iterator by default
                ext_arr = self.exterior.array(mfi).to_xp()
                sdf_arr = sdf.array(mfi).to_xp()
                ext_valid = get_valid_region(ext_arr,ng_ext)
                sdf_valid = get_valid_region(sdf_arr,ng_sdf)
                ext_valid += (sdf_valid <= 0.0).astype(ext_valid.dtype) # can detect overlapping sinks
                # np.maximum(ext_arr, sink_mask, out=ext_arr)       # only from 0 to 1
        
        self.exterior.fill_boundary()
    
    def _bind_method(self):
        if self.method == "areaWeighting":
            self._deposit_charge = self.area_weighting
        elif self.method == "surfaceWeighting":
            self._deposit_charge = self.surface_weighting
            self._initialize_exterior_mask()   
        else:
            self._deposit_charge = self.warpx_deposit_charge
    
    def deposit_charge(self,):
        self._rho.set_val(0.0)
        self._deposit_charge()
        self._rho.sum_boundary(self.geom.periodicity())
        # self._rho.fill_boundary(self.geom.periodicity()) # not needed?
        if self.persistent_charge:
            self.rho.saxpy(1.0, self._rho, 0, 0, 1, 0)
            
    def warpx_deposit_charge(self,):
        #mpit.mpi_print(f"shape: {self.rho.shape}",ranks="all")
        for species_pc in self.species_pc:
            species_pc.deposit_charge(self._rho,lev=self.lev)
            
    def area_weighting(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                self._area_weighting(pti,rho_arr, q, self.lev)
                
    def surface_weighting(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            # nudge = ("nx" in species_pc.real_soa_names) and ("nz" in species_pc.real_soa_names)
            nudge = 0.5
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                ext_arr = self.exterior.array(pti).to_xp()
                self._surface_weighting(pti,rho_arr,ext_arr, q, nudge, self.lev)
                # self._area_weighting(pti,rho_arr, q,  self.lev)

    def _area_weighting(self,pti,rho_arr,q,lev=0):
        """
        Deposits charge onto a 2D numpy grid 'rho' [nx, nz].
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
        # mpit.mpi_print(f"pti index {pti.index}: min max:{min(z):.5} {max(z):.5}","all")
        dx, dz = self.dxyz
        xmin, zmin = self.lower_bound
        ng_x = self._rho.n_grow_vect[0]
        ng_z = self._rho.n_grow_vect[1]

        # Get the global cell index offset for this specific grid box
        fab_lo = pti.validbox().small_end
        fab_lo_x, fab_lo_z = fab_lo[0], fab_lo[1]
        
        #mpit.mpi_print(f"pti index {pti.index}: rho shape {rho_arr.shape}","all")
        #mpit.mpi_print(f"pti index {pti.index}: lo_z {fab_lo_z}","all")
        # mpit.mpi_print(f"pti index {pti.index}: ng_z {ng_z}","all")
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
    
    def _surface_weighting(self,pti,rho_arr,ext_arr,q,nudge=0.0,lev=0):
        """
        Deposits charge exclusively into dielectric nodes.
        - nx, ny: Normal unit vectors pointing INTO the dielectric.
        - is_dielectric: 2D boolean mask (True = Dielectric, False = Vacuum).
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
        dx, dz = self.dxyz
        xmin, zmin = self.lower_bound
        
        # Nudge scraped particle positions slightly into dielectric along surface normal
        if nudge > 0:
            nx, nz = -pti["nx"], -pti["nz"]
            dnudge = nudge * min(dx, dz)
            x = x + dnudge * nx
            z = z + dnudge * nz

        ng_x = self._rho.n_grow_vect[0]
        ng_z = self._rho.n_grow_vect[1]

        # Get the global cell index offset for this specific grid box
        fab_lo = pti.validbox().small_end
        fab_lo_x, fab_lo_z = fab_lo[0], fab_lo[1]
        
        # Convert particle positions to local array indices (accounting for offset + ghost cells)
        x_cell = (x - xmin) / dx - fab_lo_x + ng_x
        z_cell = (z - zmin) / dz - fab_lo_z + ng_z

        i = self.xp.floor(x_cell).astype(int)
        j = self.xp.floor(z_cell).astype(int)

        fx, fz = x_cell - i, z_cell - j
        
        # Standard bilinear shape factors
        s00 = (1.0 - fx) * (1.0 - fz)
        s10 = fx * (1.0 - fz)
        s01 = (1.0 - fx) * fz
        s11 = fx * fz
        
        # Dielectric mask at corner nodes
        ext_view = ext_arr.squeeze()
        m00 = ext_view[i, j].astype(float)
        m10 = ext_view[i + 1, j].astype(float)
        m01 = ext_view[i, j + 1].astype(float)
        m11 = ext_view[i + 1, j + 1].astype(float)
    
        # Apply mask and compute normalization denominator
        w00 = s00 * m00
        w10 = s10 * m10
        w01 = s01 * m01
        w11 = s11 * m11
        
        w_tot = w00 + w10 + w01 + w11
        # mpit.mpi_print("AFTER weight mask apply: surface_weighting()")
        
        cell_area = dx * dz
        q_eff = (q * w) / cell_area
        
        # Safe division: zero out weights where no dielectric node is in stencil
        valid = w_tot > 0

        w_tot_safe = self.xp.where(valid, w_tot, 1.0)

        w00 = self.xp.where(valid, (w00 / w_tot_safe) * q_eff, 0.0)
        w10 = self.xp.where(valid, (w10 / w_tot_safe) * q_eff, 0.0)
        w01 = self.xp.where(valid, (w01 / w_tot_safe) * q_eff, 0.0)
        w11 = self.xp.where(valid, (w11 / w_tot_safe) * q_eff, 0.0)
        
        # Slice 4D array (nx, nz, 1, 1) down to 2D
        fab_view = rho_arr.squeeze()
        # Deposit into dielectric nodes
        self.xp.add.at(fab_view, (i, j), w00)
        self.xp.add.at(fab_view, (i + 1, j), w10)
        self.xp.add.at(fab_view, (i, j + 1), w01)
        self.xp.add.at(fab_view, (i + 1, j + 1), w11)
        
        if self.debug:
            mpit.mpi_print("##########  Step %i  #######"%self.sim.extension.warpx.getistep(lev))
            num_lost = len(valid) - valid.sum()
            Qin = w.sum() * q / cell_area
            Qout = w00.sum()+w10.sum()+w01.sum()+w11.sum()
            s00 = s00 * q * w / cell_area
            s10 = s10 * q * w / cell_area
            s01 = s01 * q * w / cell_area
            s11 = s11 * q * w / cell_area
            mpit.mpi_print("s00, s10, s01, s11 = %.2e, %.2e, %.2e, %.2e,"%(s00.sum(),s10.sum(),s01.sum(),s11.sum()))
            mpit.mpi_print("w00, w10, w01, w11 = %.2e, %.2e, %.2e, %.2e,"%(w00.sum(),w10.sum(),w01.sum(),w11.sum()))
            mpit.mpi_print("Qout/Qin = %.3f"%(Qout/Qin))
            mpit.mpi_print("Lost Particles = %i"%(num_lost))
        

        
    def get_pc_data(self, species_pc, variables, lev=0):
        data_arrays = species_pc.get_particle_real_arrays('x', lev)
        return tuple(self.concat(data_arrays) for var in variables)
    
    def _create_scaler_field(self,name,refField,exist_ok=True):
        if self.sim.fields.has(name,self.lev):
            if exist_ok:
                return self.sim.fields.get(name,level=refField.level)
            else:
                raise Exception("Field '{name}' already exists and 'exist_ok' is False")
        
        return self.sim.fields.alloc_init(name=name,
                            level=refField.level,
                            ba=refField.box_array(),
                            dm=refField.dm(),
                            ncomp=refField.n_comp,
                            ngrow=refField.n_grow_vect,
                            initial_value=0.,
                            redistribute=True,
                            redistribute_on_remake=True,
                            checkpoint_restart=False)
        
    
        