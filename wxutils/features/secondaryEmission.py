# -*- coding: utf-8 -*-
# import copy
# import time

import os
from pathlib import Path
from mpi4py import MPI as mpi

import numpy as np
from pywarpx import particle_containers, picmi,callbacks
from pywarpx.LoadThirdParty import load_cupy

from wxutils.ops.physics import energy_to_velocity
from wxutils.features.helpers import set_species_params
# from wxutils.debug import check_array
from wxutils.utils import to_cpu
from wxutils.core import CallbackBase
from wxutils.features.deposit import Deposit
from wxutils.diag.store import Diagnostic1D,save_to_npy

constants = picmi.constants

saveloc = Path('./diags/hist')
class SecondaryEmission(CallbackBase):
    def __init__(self,species0,sigmaMax,Emax,boundary="eb",**kw):
        self.boundary = boundary
        self.species0 = set_species_params(species0,boundary)
        self.species1 = kw.pop("species1", None)
        self.sigmaMax = sigmaMax
        self.Emax = Emax
        self.Emin = kw.pop("Emin", 5.0)
        self.Eemit = kw.pop("Eemit", 5.0)
        self.mask = kw.pop("mask",None)
        self.dumprate = kw.pop("dumprate",None)
        self.rhoSurfName = kw.pop("rhoSurf","rho_surf")
        self.wThresh = kw.pop("wThresh",1e-5)
        self.backend = kw.pop("backend","cupy") # 'cupy' will fallback to numpy if on cpu
        self.dep_method = kw.pop("dep_method","warpx")
        self.dep_sink = kw.pop("dep_sink",None)
        
        self.debug = kw.pop("debug",None)
        
        if self.species1 is None:
            self.species1 = self.species0
        else:
            self.species1 = set_species_params(self.species1)
        
        
        
        self.Uemit = energy_to_velocity(self.Eemit)
        self.saveloc = "./diags/fields/"
        self.rank = mpi.COMM_WORLD.Get_rank()
        self.lev = 0
        
        
        
    def pre_initialize(self,sim):
        self.sim = sim
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installafterstep(self.gen_secondary)
        callbacks.installcallback("afterdeposition",self.add_surface_charge)
        extra_attrs = {"nx":0.0,"nz":0.0}
        self.surface_species0 = picmi.Species(name="surface_species0",
                                        mass = self.species0.mass,
                                        charge = self.species0.charge,
                                        warpx_save_particles_at_eb=False,
                                        warpx_add_real_attributes = extra_attrs,
                                        )
        self.surface_species1 = picmi.Species(name="surface_species1",
                                        mass = self.species1.mass,
                                        charge = -self.species1.charge,
                                        warpx_save_particles_at_eb=False,
                                        warpx_add_real_attributes = extra_attrs,
                                        )
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=[1,1], grid=self.sim.solver.grid)
        sim.add_species(self.surface_species0,layout=layout)
        sim.add_species(self.surface_species1,layout=layout)
        self.deposit = Deposit(sim=self.sim,
                               rho=self.rhoSurfName,
                               species=[self.surface_species0,self.surface_species1],
                               method=self.dep_method,
                               sink=self.dep_sink,
                               persistent_charge=True)
    
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self.sey = NumericalSEY(self.Emax,self.Emin,backend=self.backend)
        self.get_impact_energy = ImpactEnergyCalculator(self.species0.mass,backend=self.backend)
        self.rhoSurf = self.initialize_surface_rho(self.rhoSurfName)
        self.rho = self.sim.fields.get("rho_fp",level=self.lev)
        self.species1_pc = self.sim.particles.get(self.species1.name)
        self.surface_species0_pc = self.sim.particles.get(self.surface_species0.name)
        self.surface_species1_pc = self.sim.particles.get(self.surface_species1.name)
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()
        self.bufferVars = ["x","z","ux","uz","nx","nz","w","deltaTimeScraped"]
        self.setup_diagnostics()
        if self.dumprate is not None:
            os.makedirs(self.saveloc,exist_ok=True)
            self.dumprate = int(self.dumprate)
            
    def initialize_surface_rho(self,name="rho_surf"):
        if self.sim.fields.has(name,self.lev):
            return
        rho = self.sim.fields.get('rho_fp',level=self.lev)
        self.rhoSurf = self.sim.fields.alloc_init(name=name,
                            level=self.lev,
                            ba=rho.box_array(),
                            dm=rho.dm(),
                            ncomp=rho.n_comp,
                            ngrow=rho.n_grow_vect,
                            initial_value=0.,
                            redistribute=True,
                            redistribute_on_remake=True,
                            checkpoint_restart=False)
        return self.rhoSurf

    def _gen_secondary(self,x,z,y,ux,uz,uy,nx,nz,ny,w,delta_t):
        eng = self.get_impact_energy(ux,uz)
        nSec = self.sigmaMax*self.sey(eng)
        
        wSec = w*nSec
        I = (wSec>self.wThresh)
        wSec = to_cpu(wSec[I])
        uxSec = self.Uemit * nx[I]
        uzSec = self.Uemit * nz[I]
        tr = self.sim.time_step_size - delta_t[I]
        
        xSec = to_cpu(x[I] + tr * uxSec)
        zSec = to_cpu(z[I] + tr * uzSec)
        
        self.species1_pc.add_particles(
            x=xSec,
            z=zSec,
            #y=y[I],
            ux=to_cpu(uxSec),
            uz=to_cpu(uzSec),
            w=wSec,
            unique_particles=True
        )  
        
        # Determine charge
        self.surface_species0_pc.add_particles(x=to_cpu(x),z=to_cpu(z),w=to_cpu(w),nx=to_cpu(nx),nz=to_cpu(nz) )  
        self.surface_species1_pc.add_particles(x=to_cpu(x[I]),z=to_cpu(z[I]),w=wSec,nx=to_cpu(nx[I]),nz=to_cpu(nz[I]))
        self.deposit.deposit_charge()
        self.surface_species0_pc.clear_particles()
        self.surface_species1_pc.clear_particles()
        
        if self.debug:
            # self.avg_sey.log(nSec.sum(),self.sim.extension.warpx.gett_new(self.lev),count=nSec.size)
            print(f"N Secondaries Avg: {nSec.mean()}")

    def gen_secondary(self):
        name = self.species0.name
        boundary = self.boundary
        x,z,ux,uz,nx,nz,w,delta_t = self.get_buffer_data(name,boundary,variables=self.bufferVars)
        I = self.mask(x, z) if self.mask is not None else True
        self._gen_secondary(x[I], z[I], None, ux[I], uz[I], None, nx[I], nz[I], None, w[I], delta_t[I])
        
    def add_surface_charge(self):
        self.rho.saxpy(1.0, self.rhoSurf, 0, 0, 1, 0)
        self.save_field(self.rho,"rho_total")
        self.save_field(self.rhoSurf,"rho_surf")
    
    def save_field(self,field,savename):
        if self.dumprate is not None:
            i = self.sim.extension.warpx.getistep(0)
            if i % self.dumprate == 0:
                path = f"{self.saveloc}/{savename}_{i}.npy"
                save_to_npy(field,path)
                
    def get_buffer_data(self, species_name, boundary, variables, lev=0):
        return tuple(
            self.concat(self.buffer.get_particle_scraped_this_step(species_name, boundary, var, lev))
            for var in variables
        )
    
    def setup_diagnostics(self):
        if self.debug:
            path = saveloc /Path("avgSEY")
            self.avg_sey = Diagnostic1D(path,interval=self.dumprate,reduce_op="mean")

###### secondary emission models
# NOTE: numba is slower than numpy at this point in the hop funnel
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Dummy decorator if Numba is missing."""
        return lambda fn: fn
  
class NumericalSEY:
    def __init__(self, Emax, Eth, backend="auto"):
        """
        Calculates secondary electron emission yield (SEY).
        Binds execution targets during __init__ for zero-branch runtime calls.
        """
        self.xp, _ = load_cupy()
        self.HAS_CUPY = self.xp.__name__ == 'cupy'
        self.Emax = float(Emax)
        self.Eth = float(Eth)

        # 1. Resolve active engine once during instantiation
        requested = backend.lower()
        if requested == 'auto':
            if self.HAS_CUPY:
                self.backend = 'cupy'
            elif HAS_NUMBA:
                self.backend = 'numba'
            else:
                self.backend = 'numpy'
        else:
            self.backend = requested

        # 2. Bind the runner function handle to the instance
        self._bind_backend()

    def _bind_backend(self):
        # --- GPU PATH (CuPy) ---
        if self.backend == 'cupy' and self.HAS_CUPY:
            cuda_code = f"""
            T xs = (x - (T){self.Eth}) / ((T){self.Emax} - (T){self.Eth});
            if (xs <= (T)0.0) {{
                out = (T)0.0;
            }} else if (xs <= (T)1.0) {{
                out = pow(xs * exp((T)1.0 - xs), (T)0.56);
            }} else if (xs <= (T)3.6) {{
                out = pow(xs * exp((T)1.0 - xs), (T)0.25);
            }} else {{
                out = (T)1.125 / pow(xs, (T)0.35);
            }}
            """
            self._gpu_kernel = self.xp.ElementwiseKernel('T x', 'T out', cuda_code, 'gpu_kernel')
            self._run = self._cupy_run
        elif self.backend == 'numba' and HAS_NUMBA:
            self._run = self._numba_run
        else:
            self._run = self._numpy_run

    # --- Engine Handlers ---
    def _cupy_run(self, x):
        out = self.xp.empty_like(x)
        self._gpu_kernel(x, out)
        return out

    def _numba_run(self, x):
        return self._cpu_kernel(x, self.Emax, self.Eth)

    def _numpy_run(self, x):
        out = np.zeros_like(x, dtype=np.float64)
        xs = (x - self.Eth) / (self.Emax - self.Eth)

        m1 = (xs > 0.0) & (xs <= 1.0)
        m2 = (xs > 1.0) & (xs <= 3.6)
        m3 = xs > 3.6

        x1, x2, x3 = xs[m1], xs[m2], xs[m3]

        out[m1] = (x1 * np.exp(1.0 - x1)) ** 0.56
        out[m2] = (x2 * np.exp(1.0 - x2)) ** 0.25
        out[m3] = 1.125 / (x3 ** 0.35)

        return out

    @staticmethod
    @njit(fastmath=True)
    def _cpu_kernel(x, Emax, Eth):
        out = np.empty_like(x)
        inv_range = 1.0 / (Emax - Eth)

        for i in range(x.shape[0]):
            xs = (x[i] - Eth) * inv_range
            if xs <= 0.0:
                out[i] = 0.0
            elif xs <= 1.0:
                out[i] = (xs * np.exp(1.0 - xs)) ** 0.56
            elif xs <= 3.6:
                out[i] = (xs * np.exp(1.0 - xs)) ** 0.25
            else:
                out[i] = 1.125 / (xs**0.35)

        return out

    def __call__(self, x):
        return self._run(x)


class ImpactEnergyCalculator:
    def __init__(self, mass, relativistic=True,backend='auto'):
        self.xp, _ = load_cupy()
        self.HAS_CUPY = self.xp.__name__ == 'cupy'
        self.backend = backend
        # Pre-compute physical scalar factors to avoid runtime math overhead
        self.m_over_q = float(mass / constants.q_e)
        self.inv_c_sq = float(1.0 / (constants.c**2))
        self.k_nonrel = float(0.5 * mass / constants.q_e)
        self.relativistic = bool(relativistic)

        # 1. Resolve active engine once during instantiation
        requested = backend.lower()
        if requested == 'auto':
            if self.HAS_CUPY:
                self.backend = 'cupy'
            elif HAS_NUMBA:
                self.backend = 'numba'
            else:
                self.backend = 'numpy'
        else:
            self.backend = requested

        # 2. Bind the exact function references to the instance
        self._bind_backend()

    def _bind_backend(self):
        # --- GPU PATH (CuPy) ---
        if self.backend == 'cupy' and self.HAS_CUPY:
            cuda_2d = f"""
            T u_sq = ux * ux + uz * uz;
            if ({int(self.relativistic)}) {{
                T u_norm_sq = u_sq * (T){self.inv_c_sq};
                out = (T){self.m_over_q} * u_sq / (sqrt((T)1.0 + u_norm_sq) + (T)1.0);
            }} else {{
                out = (T){self.k_nonrel} * u_sq;
            }}
            """
            cuda_3d = f"""
            T u_sq = ux * ux + uy * uy + uz * uz;
            if ({int(self.relativistic)}) {{
                T u_norm_sq = u_sq * (T){self.inv_c_sq};
                out = (T){self.m_over_q} * u_sq / (sqrt((T)1.0 + u_norm_sq) + (T)1.0);
            }} else {{
                out = (T){self.k_nonrel} * u_sq;
            }}
            """
            self._gpu_2d = self.xp.ElementwiseKernel('T ux, T uz', 'T out', cuda_2d, 'impact_energy_2d_gpu')
            self._gpu_3d = self.xp.ElementwiseKernel('T ux, T uy, T uz', 'T out', cuda_3d, 'impact_energy_3d_gpu')

            self._run_2d = self._cupy_2d
            self._run_3d = self._cupy_3d

        elif self.backend == 'numba' and HAS_NUMBA:
            self._run_2d = self._numba_wrapper_2d
            self._run_3d = self._numba_wrapper_3d
        else:
            self._run_2d = self._numpy_2d
            self._run_3d = self._numpy_3d

    # --- CuPy Handlers ---
    def _cupy_2d(self, ux, uz):
        out = self.xp.empty_like(ux)
        self._gpu_2d(ux, uz, out)
        return out

    def _cupy_3d(self, ux, uy, uz):
        out = self.xp.empty_like(ux)
        self._gpu_3d(ux, uy, uz, out)
        return out

    # --- Numba Kernels & Wrappers ---
    @staticmethod
    @njit(fastmath=True)
    def _numba_kernel_2d(ux, uz, m_over_q, inv_c_sq, k_nonrel, relativistic):
        out = np.empty_like(ux)
        for i in range(ux.shape[0]):
            u_sq = ux[i] * ux[i] + uz[i] * uz[i]
            if relativistic:
                u_norm_sq = u_sq * inv_c_sq
                out[i] = m_over_q * u_sq / (np.sqrt(1.0 + u_norm_sq) + 1.0)
            else:
                out[i] = k_nonrel * u_sq
        return out

    @staticmethod
    @njit(fastmath=True)
    def _numba_kernel_3d(ux, uy, uz, m_over_q, inv_c_sq, k_nonrel, relativistic):
        out = np.empty_like(ux)
        for i in range(ux.shape[0]):
            u_sq = ux[i] * ux[i] + uy[i] * uy[i] + uz[i] * uz[i]
            if relativistic:
                u_norm_sq = u_sq * inv_c_sq
                out[i] = m_over_q * u_sq / (np.sqrt(1.0 + u_norm_sq) + 1.0)
            else:
                out[i] = k_nonrel * u_sq
        return out

    def _numba_wrapper_2d(self, ux, uz):
        return self._numba_kernel_2d(ux, uz, self.m_over_q, self.inv_c_sq, self.k_nonrel, self.relativistic)

    def _numba_wrapper_3d(self, ux, uy, uz):
        return self._numba_kernel_3d(ux, uy, uz, self.m_over_q, self.inv_c_sq, self.k_nonrel, self.relativistic)

    # --- Pure NumPy Fallbacks ---
    def _numpy_2d(self, ux, uz):
        u_sq = ux**2 + uz**2
        if self.relativistic:
            u_norm_sq = u_sq * self.inv_c_sq
            return self.m_over_q * u_sq / (np.sqrt(1.0 + u_norm_sq) + 1.0)
        return self.k_nonrel * u_sq

    def _numpy_3d(self, ux, uy, uz):
        u_sq = ux**2 + uy**2 + uz**2
        if self.relativistic:
            u_norm_sq = u_sq * self.inv_c_sq
            return self.m_over_q * u_sq / (np.sqrt(1.0 + u_norm_sq) + 1.0)
        return self.k_nonrel * u_sq

    # --- Clean Runtime Entry Point ---
    def __call__(self, ux, uz, uy=None):
        if uy is None:
            return self._run_2d(ux, uz)
        return self._run_3d(ux, uy, uz)

