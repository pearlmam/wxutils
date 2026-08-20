import os
import atexit
from pathlib import Path
import h5py
import numpy as np
from pywarpx.LoadThirdParty import load_cupy
from wxutils.utils import mpi_enabled
if mpi_enabled:
    from mpi4py import MPI as mpi
else:
    mpi = None


class Diagnostic1D:
    def __init__(self, path,nsteps=None, interval=None, filetype="h5", datatype=None):
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
        self.data = self.xp.zeros((self.buf_size, 2), dtype=self.xp.float64)
        
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
            
        if self.rank == 0:
            os.makedirs(self.path.parent, exist_ok=True)
            self.initialize()
        
        atexit.register(self.on_exit)

    def initialize(self):
        """Creates or resets the HDF5 file with VizSchema metadata and empty resizable datasets (Rank 0 only)."""
        if self.rank != 0:
            return

        with h5py.File(str(self.path), "w") as h5f:
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
        self.data[self.buf_step, 0] = x0
        self.data[self.buf_step, 1] = x1
        self.buf_step += 1
        
        if self.buf_step >= self.buf_size:
            self.save()
          
    def save(self):
        """Collective flush—reduces entire buffered vector across ranks at once."""
        if self.buf_step == 0:
            return

        # 1. Pull active local slice to CPU host memory
        active_data = self.data[:self.buf_step]
        if hasattr(active_data, "get"):
            active_data = active_data.get()

        x0_local = active_data[:, 0]
        x1_local = active_data[:, 1]

        # 2. Perform chunked vector reduction across MPI ranks
        if self.size > 1 and self.reduce_op is not None:
            x1_global = self.comm.reduce(x1_local, op=self.reduce_op, root=0)
        else:
            x1_global = x1_local

        # 3. Only Rank 0 writes the aggregated array to disk
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

        # 4. Reset local buffer index on all ranks
        self.buf_step = 0
        
    def on_exit(self):
        self.save()