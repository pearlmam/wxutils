import os
import atexit
from pathlib import Path
import h5py
import numpy as np
from importlib.metadata import version
from pywarpx.LoadThirdParty import load_cupy
import wxutils.mpitools as mpit
import openpmd_api as opmd
import getpass
from pywarpx import callbacks

saveloc = Path('./diags/fields')


class DiagnosticBase:
    def __init__(self, path):
        self.xp, _ = load_cupy()
        self.path = Path(path)
        
    def grid_info(self):
        if not hasattr(self,'sim'):
            raise AttributeError(f"class {self.__class__.__name__} must be run pre_initialize() before grid_info() can be called.")
        self.dims = self.sim.solver.grid.number_of_dimensions
        self.number_of_cells = self.sim.solver.grid.number_of_cells
        self.lower_bound = self.sim.solver.grid.lower_bound
        self.upper_bound = self.sim.solver.grid.upper_bound
        self.dxyz = self.xp.array((self.xp.array(self.upper_bound)-self.xp.array(self.lower_bound))/self.xp.array(self.number_of_cells))
        
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
        
    
def save_to_npy(field,path,suffix=None):
    global_data = field[...]   # this does an mpi gather on all ranks
    if mpit.get_rank() == 0:
        np.save(path, global_data)


class Diagnostic2D(DiagnosticBase):
    def __init__(self, name,period,path=None):
        
        self.suffix_format = "%06T"
        self.path = saveloc /Path(name) /f"openpmd_{self.suffix_format}.bp5"
        self.name = name
        self.period = period
        self.lev = 0
        
        
        
    def pre_initialize(self,sim):
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installafterstep(self.save)
        callbacks.installcallback("onbreaksignal",self.close)
        self.sim = sim
        self.series = opmd.Series(self.path, opmd.Access.create)
        self.series.author = getpass.getuser()
        self.series.set_attribute("dependencies", f"{self.series.software} {self.series.software_version}")
        self.series.set_software('wxutils',version("wxutils"))
    
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self.grid_info()
        self.field_data = self.sim.fields.get(self.name,level=self.lev)
        os.makedirs(self.path.parent,exist_ok=True)
        with open(self.path.parent/"paraview.pmd", "w") as f:
            f.write(f"openpmd_{self.suffix_format}.bp5\n")
        
    def save(self):
        step = self.sim.extension.warpx.getistep(self.lev)
        if step % self.period == 0:
            data = self.field_data[...]
            shape = list(data.shape)
            # shape.append(1)
            data = data.reshape(shape[0], 1, shape[1])
            shape = data.shape
            step = self.sim.extension.warpx.getistep(self.lev)
            t = self.sim.extension.warpx.gett_new(self.lev)
            
            it = self.series.iterations[step]
            #print(dir(it))
            it.set_time(t)
            it.set_dt(self.sim.time_step_size)
            it.set_time_unit_SI(1.0)
            # it.set_attribute("data_time",t)
            mesh = it.meshes[self.name]
            mesh.grid_spacing = [self.dxyz[0],1.0,self.dxyz[1]]
            mesh.grid_global_offset = [0., 0., 0.]
            mesh.axis_labels = ["x","y","z"]
            mesh.data_order = 'C'
            component = mesh[opmd.Mesh.SCALAR]
            print(f"dxz={mesh.grid_spacing}, offset = {mesh.grid_global_offset},shape = {data.shape}")
            component.reset_dataset(opmd.Dataset(data.dtype, shape))
            component[:, :] = data
            it.close()
            self.series.flush()
    
    def close(self,):
        self.series.flush()
        self.series.close()
        
        
        
def save_to_opmd(field,path,fieldname='field_data',step=0):
    series = opmd.Series(path, opmd.Access.create)
    series.author = getpass.getuser()
    series.set_attribute("dependencies", f"{series.software} {series.software_version}")
    series.set_software('wxutils',version("wxutils"))
    
    it = series.iterations[step]
    it.time = 0.0
    
    
    
    mesh = it.meshes[fieldname]
    component = mesh["var"]
    dataset = opmd.Dataset(field.dtype, field.shape)
    component.reset_dataset(dataset)
    component[:, :] = field
    series.flush()
    series.close()
    

    
if __name__ == "__main__":
    
    data = np.arange(150 * 300,dtype=np.float64).reshape(150, 300)
    suffix_format = "%05T"
    path = Path(f"openpmd_{suffix_format}.bp5")
    fieldname = 'var'
    save_to_opmd(data,path,fieldname,0)
    with open(path.parent/"paraview.pmd", "w") as f:
        f.write(f"openpmd{suffix_format}.bp5")
    
    