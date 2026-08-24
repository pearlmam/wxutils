import os
import atexit
from pathlib import Path
import h5py
import numpy as np
from importlib.metadata import version
from pywarpx.LoadThirdParty import load_cupy
from wxutils.utils import mpi_enabled,get_rank,mpiprint

if mpi_enabled:
    from mpi4py import MPI as mpi
else:
    mpi = None

class DiagnosticBase:
    def __init__(self):
        self.xp, _ = load_cupy()
    
    def mpi_info(self):
        if mpi_enabled:
            self.comm = mpi.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.reduce_op = mpi.SUM
        else:
            self.comm = None
            self.rank = 0
            self.size = None
            self.reduce_op = None


class Diagnostic1D(DiagnosticBase):
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
        self.mpi_info()
        
            
        if self.rank == 0:
            os.makedirs(self.path.parent, exist_ok=True)
            self.initialize()
        
        atexit.register(self.on_exit)

    def initialize(self):
        """Creates or resets the HDF5 file with VizSchema metadata and empty resizable datasets (Rank 0 only)."""
        if self.rank != 0:
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
        if self.size > 1 and self.reduce_op is not None:
            x1_global = np.empty_like(x1_local) if self.rank == 0 else None
            self.comm.Reduce(x1_local, x1_global, op=self.reduce_op, root=0)
        else:
            x1_global = x1_local
    
        # 3. Only Rank 0 writes to disk
        if self.rank == 0:
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
        
        
class DiagnosticField(DiagnosticBase):
    def __init__(self,backend):
        self.backend=backend
        
    
def save_to_npy(field,path,suffix=None):
    global_data = field[...]   # this does an mpi gather on all ranks
    if get_rank() == 0:
        np.save(path, global_data)

def save_to_plotfile(field,path,suffix=None):
    raise NotImplementedError(f"save_to_plotfile not implemented")
    
def save_to_h5(field,name='var',path='./',suffix=None):
    raise NotImplementedError(f"save_to_h5 not implemented")
    global_shape = field.shape
    local_data = field[...]
    local_slice = None
    comm = mpi.COMM_WORLD
    with h5py.File(path, "w", driver="mpio", comm=comm) as f:
        # Set global dataset shape and write rank-specific slices
        dset = f.create_dataset(name, shape=global_shape, dtype=local_data.dtype)
        dset[local_slice] = local_data
    
    
    