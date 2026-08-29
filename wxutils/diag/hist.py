# -*- coding: utf-8 -*-
import numpy as np

from pywarpx import particle_containers, picmi,callbacks,libwarpx
from wxutils.features.helpers import set_species_params

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
    
        self.speciesList = [set_species_params(_species,self.boundaryList) for _species in self.speciesList]
        self.lev = 0
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        
    def post_initialize(self):
        super().post_initialize()
        self.data = self.xp.zeros((2, self.save_period), dtype=self.xp.float64)
        # self.species_pc = [self.sim.particles.get(species.name) for species in self.speciesList]
        self.buffer = particle_containers.ParticleBoundaryBufferWrapper()

        if self.save_period is None:
            # check for diagnostics and grab that value
            if len(self.sim.diagnostics)>0: 
                self.save_period = self.sim.diagnostics[0].period
    
    
    def get(self):
        '''must return tuple of (value,time), this will get logged'''
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
        return q/dt,t
    
    def log(self):
        x0,x1 = self.get()

        self.data[0, self.buf_step] = x0
        self.data[1, self.buf_step] = x1
        self.buf_step += 1
        super().log()
    
    def save(self):
        """Collective flush—reduces entire buffered vector across ranks at once."""
        if self.buf_step == 0:
            return
    
        # 1. Pull active local slice to CPU host memory
        active_data = self.data
        if hasattr(active_data, "get"):
            active_data = active_data.get()
    
        # Rows are inherently C-contiguous in memory
        x1_local = active_data[1]
        
        # 2. Perform chunked vector reduction across MPI ranks
        if self.mpii.enabled and self.reduce_op is not None:
            x1_global = np.empty_like(x1_local) if self.mpi.is_root else None
            self.mpii.comm.Reduce(x1_local, x1_global, op=self.reduce_op, root=0)
        else:
            x1_global = x1_local
        
        super().save()
        self.buf_step = 0
        
        
class VizSchema1D():
    def __init__(self, path,nsteps=None, interval=None, filetype="h5", datatype=None,reduce_op=None):
        self.xp, _ = load_cupy()

        self.path = Path(path)
        self.name = self.path.stem
        self.timeGroupName = f"timeSeries{self.name}"
        self.filetype = filetype
        if nsteps is None and interval is None:
            raise TypeError("Undeveloped feature: number of steps or interval must be defined")
        self.nsteps = int(nsteps) if nsteps is not None else nsteps
        self.interval = int(interval) if interval is not None else interval
        
        if self.path.suffix == '':
            self.path = self.path.with_suffix(f".{filetype}")
        
        # Buffer sized only for flush interval
        self.buf_size = self.interval if self.interval else self.nsteps
        self.buf_step = 0
        self.data = self.xp.zeros((2, self.buf_size), dtype=self.xp.float64)

        if mpit.enabled():
            self.reduce_op = mpit.mpi.SUM
        else:
            self.reduce_op = None
            
        if mpit.get_rank() == 0:
            os.makedirs(self.path.parent, exist_ok=True)
            self.initialize()
        
        atexit.register(self.on_exit)

    def initialize(self):
        """Creates or resets the HDF5 file with VizSchema metadata and empty resizable datasets (Rank 0 only)."""
        if mpit.get_rank() != 0:
            return

        with h5py.File(str(self.path), "w") as h5f:
            mesh_group = h5f.create_group("runInfo")
            mesh_group.attrs["software"] = np.bytes_("wxutils")
            mesh_group.attrs["version"] = np.bytes_(version("wxutils"))
            
            mesh_group = h5f.create_group(self.timeGroupName)
            mesh_group.attrs["vsType"] = np.bytes_("mesh")
            mesh_group.attrs["vsKind"] = np.bytes_("rectilinear")
            mesh_group.attrs["vsAxis0"] = np.bytes_("time")
        
            # Create empty resizable time dataset
            mesh_group.create_dataset(
                "time", 
                shape=(0,), 
                maxshape=(None,), 
                chunks=(self.buf_size,), 
                dtype=np.float64
            )
            
            # Create empty resizable variable dataset
            dset_var = h5f.create_dataset(
                self.name, 
                shape=(0,), 
                maxshape=(None,), 
                chunks=(self.buf_size,), 
                dtype=np.float64
            )
            dset_var.attrs["vsType"] = np.bytes_("variable")
            dset_var.attrs["vsMesh"] = np.bytes_(self.timeGroupName)

    def log(self, x1, x0):
        """Purely local logging—zero MPI latency or network communication per step."""
        self.data[0, self.buf_step] = x0
        self.data[1, self.buf_step] = x1
        self.buf_step += 1
        
        if self.buf_step >= self.buf_size:
            self.save()
          
    def save(self):
        """Collective flush—reduces entire buffered vector across ranks at once."""
        if self.buf_step == 0:
            return
    
        # 1. Pull active local slice to CPU host memory
        active_data = self.data[:, :self.buf_step]
        if hasattr(active_data, "get"):
            active_data = active_data.get()
    
        # Rows are inherently C-contiguous in memory
        x0_local = active_data[0]
        x1_local = active_data[1]
        
        # 2. Perform chunked vector reduction across MPI ranks
        if mpit.get_size() > 1 and self.reduce_op is not None:
            x1_global = np.empty_like(x1_local) if mpit.get_rank() == 0 else None
            mpit.get_comm().Reduce(x1_local, x1_global, op=self.reduce_op, root=0)
        else:
            x1_global = x1_local
    
        # 3. Only Rank 0 writes to disk
        if mpit.get_rank() == 0:
            n_samples = len(x0_local)
            with h5py.File(str(self.path), "a") as h5f:
                dset_time = h5f[f"{self.timeGroupName}/time"]
                dset_var = h5f[self.name]
    
                old_size = dset_time.shape[0]
                new_size = old_size + n_samples
                
                dset_time.resize((new_size,))
                dset_var.resize((new_size,))
                
                dset_time[old_size:new_size] = x0_local
                dset_var[old_size:new_size] = x1_global
    
        # 4. Reset local buffer index
        self.buf_step = 0
        
    def on_exit(self):
        self.save()