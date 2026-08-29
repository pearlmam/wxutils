# -*- coding: utf-8 -*-

from pywarpx import picmi,callbacks
from pywarpx.LoadThirdParty import load_cupy
from wxutils.core import CallbackBase
import wxutils.mpitools as mpit
from wxutils.features.helpers import get_valid_region
import numpy as np
from wxutils.core.base import GridInfo
class Deposit(CallbackBase):
    def __init__(self,**kw):
        
        self.rhoName = kw.pop("rho","rho_fp")
        self.method = kw.pop("method","areaWeighting")
        self.speciesList = kw.pop("species",None)
        self.persistent_charge = kw.pop("persistent_charge",None)
        self.nudge_n = kw.pop("nudge_n",None)
        self.split_spread = kw.pop("split_spread",None)
        self.split_weights = kw.pop("split_weights",[0.25,0.5,0.25])
        self.debug = kw.pop("debug",None)
        if not isinstance(self.speciesList,(list,tuple,picmi.Species)):
            raise TypeError("species must be a picmi.Species or list-like of picmi.Species")
        elif isinstance(self.speciesList,(picmi.Species)):
            self.speciesList = [self.speciesList]
        
        if  "surfaceWeighting" in self.method:
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
        self.grid_info = GridInfo(self.sim,self.xp)   # CallbackBase method: adds grid info to self
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
        if (("Quad" in self.method) and 
           ("Split" in self.method) and 
           (min(self.rho.n_grow_vect) < 2)):
            raise Exception(f"Method '{self.method}' requires Field '{self.rhoName}' to have >= 2 guard cells, set n_grow >= 2")
        
        # if ("surface" in self.method) and (self.nudge_n or self.split_spread):
        #     if max(self.nudge_n,self.split_spread) > min(self.rho.n_grow_vect):
        #         raise Exception(f"Method '{self.method}' requires Field {self.rhoName} to have >= max(nudge_n,split_spread) guard cells")
        
            
            for species_pc in self.species_pc:
                if (("nx" not in species_pc.real_soa_names) and 
                    ("nz" not in species_pc.real_soa_names) and
                    (self.nudge_n or self.split_spread)):
                    raise Exception(f"With 'nudge_n' or 'split_spread' defined, method '{self.method}' requires Species '{species_pc}' to have 'nx' and 'nz' attributes that describe the normal of the surface.")
        valid_methods = ["areaWeighting","areaWeightingQuad","surfaceWeighting","surfaceWeightingQuad","warpx"]
        if self.method == "areaWeighting":
            self._deposit_charge = self.area_weighting
        elif self.method == "areaWeightingQuad":
            self._deposit_charge = self.area_weighting_quad
        elif self.method == "surfaceWeighting":
            self._deposit_charge = self.surface_weighting
            self._initialize_exterior_mask()  
        elif self.method == "surfaceWeightingQuad":
            self._deposit_charge = self.surface_weighting_quad
            self._initialize_exterior_mask()  
        elif self.method == "warpx":
            self._deposit_charge = self.warpx_deposit_charge
        else:
            raise Exception(f"Method '{self.method}' must be one of {valid_methods}")
    
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
    
    def area_weighting_quad(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                self._area_weighting_quad(pti,rho_arr, q, self.lev)
                
    def surface_weighting(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                ext_arr = self.exterior.array(pti).to_xp()
                self._surface_weighting(pti,rho_arr,ext_arr, q, self.lev)
                
    def surface_weighting_quad(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                ext_arr = self.exterior.array(pti).to_xp()
                self._surface_weighting_quad(pti,rho_arr,ext_arr, q, self.lev)
    
    def surface_weighting_split(self,):
        for species,species_pc in zip(self.speciesList,self.species_pc):
            q = species.charge
            for pti in species_pc.iterator(level=self.lev):
                rho_arr = self._rho.array(pti).to_xp()
                ext_arr = self.exterior.array(pti).to_xp()
                self._surface_weighting_split(pti,rho_arr,ext_arr, q, self.lev)
    

    def _area_weighting(self,pti,rho_arr,q,lev=0):
        """
        Deposits charge onto a 2D numpy grid 'rho' [nx, nz].
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
        # mpit.mpi_print(f"pti index {pti.index}: min max:{min(z):.5} {max(z):.5}","all")
        dx, dz = self.grid_info.dxyz
        xmin, zmin = self.grid_info.lower_bound
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
        
        if self.debug:
            dep_weights=[w00,w01,w10,w11]
            self._debug_weights(weights=w,dep_weights=dep_weights,valid=None, q=q, lev=lev)
    
    def _area_weighting_quad(self, pti, rho_arr, q, lev=0):
        """
        Un-validated and has some for-loops that can be vectorized
        Deposits charge onto a 2D grid 'rho' [nx, nz] using 2nd-order (TSC / Quadratic Spline) area weighting.
        Requires self._rho to have at least n_grow_vect >= 2.
        
        This effectively moves the charge to the nearest grid point, and deposits charge on a 9-node stencil
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
    
        dx, dz = self.grid_info.dxyz
        xmin, zmin = self.grid_info.lower_bound
        ng_x = self._rho.n_grow_vect[0]
        ng_z = self._rho.n_grow_vect[1]
    
        # Get local patch offset
        fab_lo = pti.validbox().small_end
        fab_lo_x, fab_lo_z = fab_lo[0], fab_lo[1]
    
        # Continuous cell coordinates
        x_cell = (x - xmin) / dx - fab_lo_x + ng_x
        z_cell = (z - zmin) / dz - fab_lo_z + ng_z
    
        # Nearest Grid Point (NGP) indices
        i = self.xp.floor(x_cell + 0.5).astype(int)
        j = self.xp.floor(z_cell + 0.5).astype(int)
    
        # Fractional offsets from NGP: dx_p, dz_p in [-0.5, 0.5]
        dx_p = x_cell - i
        dz_p = z_cell - j
    
        cell_area = dx * dz
        q_eff = (q * w) / cell_area
    
        # 1D Quadratic B-spline weights for offset nodes (-1, 0, +1)
        sx_m1 = 0.5 * (0.5 - dx_p)**2
        sx_0  = 0.75 - dx_p**2
        sx_p1 = 0.5 * (0.5 + dx_p)**2
    
        sz_m1 = 0.5 * (0.5 - dz_p)**2
        sz_0  = 0.75 - dz_p**2
        sz_p1 = 0.5 * (0.5 + dz_p)**2
    
        sx = [sx_m1, sx_0, sx_p1]
        sz = [sz_m1, sz_0, sz_p1]
    
        # Slice 4D array down to 2D
        fab_view = rho_arr[:, :, 0, 0] if rho_arr.ndim == 4 else rho_arr
    
        # Deposit across 3x3 stencil
        for di_idx, di in enumerate([-1, 0, 1]):
            for dj_idx, dj in enumerate([-1, 0, 1]):
                weight = sx[di_idx] * sz[dj_idx] * q_eff
                self.xp.add.at(fab_view, (i + di, j + dj), weight)
        
        # if self.debug:
        #     dep_weights=[w00,w01,w10,w11]
        #     self._debug_weights(weights=w,dep_weights=dep_weights,valid=valid, q=q, lev=lev)
    
    
    def _surface_weighting(self,pti,rho_arr,ext_arr,q,lev=0):
        """
        Deposits charge exclusively into dielectric nodes.
        - nx, ny: Normal unit vectors pointing INTO the dielectric.
        - is_dielectric: 2D boolean mask (True = Dielectric, False = Vacuum).
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
        dx, dz = self.grid_info.dxyz
        xmin, zmin = self.grid_info.lower_bound
        
        # Nudge scraped particle positions slightly into dielectric along surface normal
        if self.nudge_n or self.split_spread:
            nx, nz = -pti["nx"], -pti["nz"]
            x, z, w = self._nudge_split_particles(x, z, w, nx, nz, self.nudge_n, self.split_spread,self.split_weights)
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
        
        ## old debug code
        # s00 = s00 * q * w / cell_area
        # s10 = s10 * q * w / cell_area
        # s01 = s01 * q * w / cell_area
        # s11 = s11 * q * w / cell_area
        # mpit.mpi_print("s00, s10, s01, s11 = %.2e, %.2e, %.2e, %.2e,"%(s00.sum(),s10.sum(),s01.sum(),s11.sum()))
        # mpit.mpi_print("w00, w10, w01, w11 = %.2e, %.2e, %.2e, %.2e,"%(w00.sum(),w10.sum(),w01.sum(),w11.sum()))
        if self.debug:
            dep_weights=[w00,w01,w10,w11]
            self._debug_weights(weights=w,dep_weights=dep_weights,valid=valid, q=q, lev=lev)
          
    def _surface_weighting_quad(self, pti, rho_arr, ext_arr, q, lev=0):
        """
        Un-validated and has some for-loops that can be vectorized
        Deposits charge exclusively into dielectric nodes using 2nd-order (TSC) weighting.
        Requires self._rho and ext_arr to have at least n_grow_vect >= 2.
        """
        x, z, w = pti["x"], pti["z"], pti["w"]
        if len(x) == 0:
            return
            
        dx, dz = self.grid_info.dxyz
        xmin, zmin = self.grid_info.lower_bound
        
        # Nudge scraped particle positions into dielectric along surface normal
        # Nudge scraped particle positions slightly into dielectric along surface normal
        if self.nudge_n or self.split_spread:
            nx, nz = -pti["nx"], -pti["nz"]
            x, z, w = self._nudge_split_particles(x, z, w, nx, nz, self.nudge_n, self.split_spread,self.split_weights)
        
    
        ng_x = self._rho.n_grow_vect[0]
        ng_z = self._rho.n_grow_vect[1]
    
        # Get local patch offset
        fab_lo = pti.validbox().small_end
        fab_lo_x, fab_lo_z = fab_lo[0], fab_lo[1]
        
        # Continuous local cell coordinates
        x_cell = (x - xmin) / dx - fab_lo_x + ng_x
        z_cell = (z - zmin) / dz - fab_lo_z + ng_z
    
        # Nearest Grid Point (NGP) indices
        i = self.xp.floor(x_cell + 0.5).astype(int)
        j = self.xp.floor(z_cell + 0.5).astype(int)
    
        # Fractional offsets from NGP in [-0.5, 0.5]
        dx_p = x_cell - i
        dz_p = z_cell - j
    
        # 1D Quadratic B-spline weights for relative nodes (-1, 0, +1)
        sx_m1 = 0.5 * (0.5 - dx_p)**2
        sx_0  = 0.75 - dx_p**2
        sx_p1 = 0.5 * (0.5 + dx_p)**2
    
        sz_m1 = 0.5 * (0.5 - dz_p)**2
        sz_0  = 0.75 - dz_p**2
        sz_p1 = 0.5 * (0.5 + dz_p)**2
    
        sx = [sx_m1, sx_0, sx_p1]
        sz = [sz_m1, sz_0, sz_p1]
    
        # Views squeezed down to 2D
        ext_view = ext_arr.squeeze()
        fab_view = rho_arr.squeeze()
    
        # Step 1: Compute masked shape factors and sum denominator over 3x3 stencil
        w_masked = {}
        w_tot = self.xp.zeros_like(x, dtype=float)
    
        for di_idx, di in enumerate([-1, 0, 1]):
            for dj_idx, dj in enumerate([-1, 0, 1]):
                s_factor = sx[di_idx] * sz[dj_idx]
                mask = ext_view[i + di, j + dj].astype(float)
                
                node_w = s_factor * mask
                w_masked[(di, dj)] = node_w
                w_tot += node_w

        # Step 2: Renormalize weights to conserve charge across valid nodes
        cell_area = dx * dz
        q_eff = (q * w) / cell_area
    
        valid = w_tot > 0
        w_tot_safe = self.xp.where(valid, w_tot, 1.0)
    
        # Step 3: Deposit into grid
        for (di, dj), node_w in w_masked.items():
            w_final = self.xp.where(valid, (node_w / w_tot_safe) * q_eff, 0.0)
            self.xp.add.at(fab_view, (i + di, j + dj), w_final)
        
        # if self.debug:
        #     dep_weights=[w00,w01,w10,w11]
        #     self._debug_weights(weights=w,dep_weights=dep_weights,valid=valid, q=q, lev=lev)

    def _nudge_split_particles(self, x, z, w, nx, nz, nudge_n=None, spread_t=None,split_weights=[0.25,0.50,0.25]):
        """
        Expands N particles into 3N sub-particles smoothed tangentially along the surface.
        - nudge_n: Normal inward displacement fraction (relative to cell size).
        - spread_t: Tangential spread distance fraction (e.g., 0.5 = half cell size).
        """
        min_dx = min(self.grid_info.dxyz[0], self.grid_info.dxyz[1])
        if self.debug: 
            w_tot_0 = w.sum()
            num_particles_0 = w.size
        if nudge_n:
            d_n = nudge_n * min_dx
            x = x + d_n * nx
            z = z + d_n * nz
        if spread_t:
            d_t = spread_t * min_dx
            # Inward normal and surface tangent vectors
            # (nx, nz are already negated inward normals from pti)
            tx, tz = -nz, nx
    
            # 3-Point Binomial weights: 25% left, 50% center, 25% right
            x_left   = x - d_t * tx
            z_left   = z - d_t * tz
            w_left   = split_weights[0] * w
        
            x_center = x
            z_center = z
            w_center = split_weights[1] * w
        
            x_right  = x + d_t * tx
            z_right  = z + d_t * tz
            w_right  = split_weights[2] * w
        
            # Concatenate into 3N sub-particle arrays
            x = self.concat([x_left, x_center, x_right])
            z = self.concat([z_left, z_center, z_right])
            w = self.concat([w_left, w_center, w_right])
            
        if self.debug:
            nsteps = 1000000
            step = self.sim.extension.warpx.getistep(self.lev)
            tol = 1e-5
            w_tot_1 = w.sum()
            num_particles_1 = w.size
            w_ratio = w_tot_1 / w_tot_0 
            n_ratio = num_particles_1 / num_particles_0
            string = "##########  Deposition Split Particles Debug: Step %i  #######\n"%step
            string += f"w_out/w_in = {w_ratio:0.3f}\n"
            string += f"n_out/n_in = {n_ratio:0.3f}\n"
            if (abs(n_ratio-3.0)>tol) or (abs(w_ratio-1.0)>tol) or step%nsteps==0:
                mpit.mpi_print(string,ranks=-1)
        
        return x,z,w

    def _debug_weights(self,weights=None,dep_weights=None,valid=None, q=None, lev=0):
        if self.debug:
            nsteps = 1000000
            step = self.sim.extension.warpx.getistep(lev)
            # only print if particles lost and every n steps
            string = "##########  Deposition Weight Continuity Debug: Step %i  #######\n"%step
            # mpit.mpi_print("##########  Step %i  #######"%self.sim.extension.warpx.getistep(lev))
            num_lost = 0
            Qratio = 1.00
            tol = 1e-5
            if valid is not None:
                num_lost = len(valid) - valid.sum()
                string += "Lost Particles = %i\n"%(num_lost)
                #mpit.mpi_print("Lost Particles = %i"%(num_lost))
            if (weights is not None) and (dep_weights is not None) and q:
                cell_area = self.xp.prod(self.xp.array(self.grid_info.dxyz[:self.grid_info.dims]))
                q_eff = q  / cell_area
                Qin = weights.sum() * q_eff
                Qout = self.xp.array([w.sum() for w in dep_weights]).sum() 
                Qratio = Qout/Qin
                string += "Qout/Qin = %.10f\n"%(Qratio)
                # mpit.mpi_print("Qout/Qin = %.3f"%(Qout/Qin))
            if num_lost > 0 or (abs(Qratio-1.0)>tol) or step%nsteps==0:
                mpit.mpi_print(string,ranks=-1)

        
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
        
    
        