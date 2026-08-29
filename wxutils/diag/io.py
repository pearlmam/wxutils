import os
import atexit
from pathlib import Path
import h5py
import numpy as np
from importlib.metadata import version
from pywarpx.LoadThirdParty import load_cupy
import wxutils.mpitools as mpit
from wxutils.utils import to_cpu
import openpmd_api as opmd
import getpass
from pywarpx import callbacks

saveloc = Path('./diags/fields')




class IO():
    def __init__(self,**kw):
        self.path = Path(kw.pop("path","./diags"))
        self._initialized = False
        
    def pre_initialize(self,sim):
        if self._initialized:
            return
        self.mpii = mpit.Info()
        self._install_callbacks()
        self.sim = sim
        self.author = getpass.getuser()
        self.software = "wxutils"
        self.version = version("wxutils")
        self._initialized = False
        self.initialize_file()
    
    def _install_callbacks(self,):
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installcallback("onbreaksignal",self.close)
        
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self._initialized = True
        
    def initialize_file(self):
        raise NotImplementedError("initialize_file is not implemented. Use it to create the data space")
    
    def create_dataset(self):
        raise NotImplementedError("create_dataset is not implemented. Use it to create the specific dataset")
    
    def save(self,data):
        pass
    
    def close():
        pass
    
class VizSchema1D(IO):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.path = self.path /"history.h5"
        self.datasets = []

    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        
        
        
    def initialize_file(self):
        if self.mpii.is_root and not self._initialized:
            with h5py.File(str(self.path), "w") as h5f:
                mesh_group = h5f.create_group("runInfo")
                mesh_group.attrs["software"] = np.bytes_(self.software)
                mesh_group.attrs["version"] = np.bytes_(self.version)
            
    def create_dataset(self,name,buf_size,dtype=np.float64):
        if self.mpii.is_root:
            if name in self.datasets:
                raise Exception(f"'{name}' already in history dataset")
            self.datasets.append(name)
            with h5py.File(str(self.path), "a") as h5f:
                timeGroupName = self.get_time_group_name(name)
                mesh_group = h5f.create_group(timeGroupName)
                mesh_group.attrs["vsType"] = np.bytes_("mesh")
                mesh_group.attrs["vsKind"] = np.bytes_("rectilinear")
                mesh_group.attrs["vsAxis0"] = np.bytes_("time")
                
                mesh_group.create_dataset(
                    "time", 
                    shape=(0,), 
                    maxshape=(None,), 
                    chunks=(buf_size,), 
                    dtype=dtype
                    )
                
                # Create empty resizable variable dataset
                dset_var = h5f.create_dataset(
                    name, 
                    shape=(0,), 
                    maxshape=(None,), 
                    chunks=(buf_size,), 
                    dtype=dtype
                    )
                dset_var.attrs["vsType"] = np.bytes_("variable")
                dset_var.attrs["vsMesh"] = np.bytes_(timeGroupName)
    
    def get_time_group_name(self,name):
        return f"timeSeries{name}"
    
    def save(self,name,data):
        """Collective flush—reduces entire buffered vector across ranks at once."""
        data = to_cpu(data)  # file io on cpu
        timeGroupName = self.get_time_group_name(name)
        x0 = data[0]
        x1 = data[1]
        if self.mpii.is_root:
            n_samples = len(x0)
            with h5py.File(str(self.path), "a") as h5f:
                dset_time = h5f[f"{timeGroupName}/time"]
                dset_var = h5f[name]
    
                old_size = dset_time.shape[0]
                new_size = old_size + n_samples
                
                dset_time.resize((new_size,))
                dset_var.resize((new_size,))
                
                dset_time[old_size:new_size] = x0
                dset_var[old_size:new_size] = x1
    
    def on_exit(self):
        self.save()
    
    
        
class OpenPMD(IO):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.suffix_format = kw.pop("suffix_format","%06T")
        self.path = self.path / "diag2"
        self.file_template = f"openpmd_{self.suffix_format}.bp5"
        self.path = kw.pop("path",self.path)
        self.path = self.path / self.file_template
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        self.series = opmd.Series(self.path, opmd.Access.create)
        self.series.author = self.author
        self.series.set_attribute("dependencies", f"{self.series.software} {self.series.software_version}")
        self.series.set_software(self.software,self.version)
        
    
    def post_initialize(self):
        super().post_initialize()
    
    def initialize_file(self,):
        if self.mpii.is_root and not self._initialized:
            os.makedirs(self.path.parent,exist_ok=True)
            with open(self.path.parent/"paraview.pmd", "w") as f:
                f.write(f"openpmd_{self.suffix_format}.bp5\n")
            
    def create_dataset(self,*args,**kwargs):
        # not needed?
        pass
            
    def save(self, name, data, grid, node_to_center=True,lev=0):

        if not (self.mpii.is_root and self._initialized):
            return
        comp_pos = [0.0]
        if node_to_center:
            data = self.node_to_cell_centered(data)
            comp_pos = [0.5]
        data = to_cpu(data)  # file io on cpu

        step = self.sim.extension.warpx.getistep(lev)
        t = self.sim.extension.warpx.gett_new(lev)
        dt = self.sim.time_step_size
    
        it = self.series.iterations[step]
        it.set_time(t)
        it.set_dt(dt)
        it.set_time_unit_SI(1.0)
    
        mesh = it.meshes[name]
        
        # Configure grid metadata and reshape data if needed
        data_formatted = self.setup_mesh_geometry(mesh, grid, data)
    
        component = mesh[opmd.Mesh.SCALAR]
        component.position = comp_pos * data_formatted.ndim
        component.reset_dataset(opmd.Dataset(data_formatted.dtype, data_formatted.shape))
        full_slice = tuple(slice(None) for _ in range(data_formatted.ndim))
        component[full_slice] = data_formatted
    
        it.close()
        self.series.flush()
    
    def setup_mesh_geometry(self, mesh, grid, data):
        """
        Configures openPMD mesh metadata based on data dimensionality.
        Pads 2D (z, x) grids into 3D (z, y, x) with unit thickness along y.
        """
        ndim = data.ndim
        # TODO need RZ check here
        if ndim == 2:
            # Reshape 2D array [Nz, Nx] -> 3D array [Nz, 1, Nx]
            data_formatted = data.reshape(data.shape[0], 1, data.shape[1])
            
            # Insert unit spacing and y-axis label for 3D representation
            dx = grid.dxyz[0] if grid.dims > 0 else 1.0
            dz = grid.dxyz[1] if grid.dims > 1 else 1.0
            mesh.grid_spacing = [dx, 1.0, dz]
            mesh.grid_global_offset = [0.0, 0.0, 0.0]
    
            if len(grid.axis_labels) == 2:
                mesh.axis_labels = [grid.axis_labels[0], 'y', grid.axis_labels[1]]
            else:
                mesh.axis_labels = list(grid.axis_labels)
    
        elif ndim == 3:
            data_formatted = data
            mesh.grid_spacing = list(grid.dxyz)
            mesh.grid_global_offset = [0.0, 0.0, 0.0]
            mesh.axis_labels = list(grid.axis_labels)
    
        elif ndim == 1:
            data_formatted = data
            mesh.grid_spacing = list(grid.dxyz)
            mesh.grid_global_offset = [0.0]
            mesh.axis_labels = list(grid.axis_labels)
    
        else:
            raise ValueError(f"Unsupported data dimension: {ndim}")
    
        mesh.data_order = 'C'
        return data_formatted
    
    def node_to_cell_centered(self,data):
        """
        Averages node-centered array data (N+1 points per axis) 
        to cell-centered data (N points per axis).
        """
        if data.ndim == 1:
            return 0.5 * (data[:-1] + data[1:])
        elif data.ndim == 2:
            return 0.25 * (
                data[:-1, :-1] + data[1:, :-1] + 
                data[:-1, 1:]  + data[1:, 1:]
            )
        elif data.ndim == 3:
            return 0.125 * (
                data[:-1, :-1, :-1] + data[1:, :-1, :-1] +
                data[:-1, 1:, :-1]  + data[1:, 1:, :-1]  +
                data[:-1, :-1, 1:]  + data[1:, :-1, 1:]  +
                data[:-1, 1:, 1:]   + data[1:, 1:, 1:]
            )
        return data
    
    def close(self,):
        self.series.flush()
        self.series.close()
        
def save_to_npy(field,path,suffix=None):
    global_data = field[...]   # this does an mpi gather on all ranks
    if mpit.get_rank() == 0:
        np.save(path, global_data)        

    
if __name__ == "__main__":
    pass
    

    
    