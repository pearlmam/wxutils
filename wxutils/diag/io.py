import os
from pathlib import Path
import h5py
import numpy as np
from importlib.metadata import version
import wxutils.mpitools as mpit
from wxutils.utils import to_cpu
import openpmd_api as opmd
import getpass

saveloc = Path('./diags/fields')

class IO():
    def __init__(self,**kw):
        self.path = Path(kw.pop("path","./diags"))
        self.parallel_write = kw.pop("parallel_write",False)
        self._initialized = False
        
    def initialize(self):
        if self._initialized:
            return
        self.mpii = mpit.Info()
        self.author = getpass.getuser()
        self.software = "wxutils"
        self.version = version("wxutils")
        self.initialize_file()
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
        
    def initialize(self):
        super().initialize()
        if self.parallel_write or self.mpii.is_root:
            self.series = opmd.Series(self.path, opmd.Access.create)
            self.series.author = self.author
            self.series.set_attribute("dependencies", f"{self.series.software} {self.series.software_version}")
            self.series.set_software(self.software,self.version)
    
    def initialize_file(self,):
        if self.mpii.is_root and not self._initialized:
            os.makedirs(self.path.parent,exist_ok=True)
            with open(self.path.parent/"paraview.pmd", "w") as f:
                f.write(f"openpmd_{self.suffix_format}.bp5\n")
            
    def create_dataset(self,*args,**kwargs):
        # not needed?
        pass
            
        
    def save(self,name,data,step,**kw):
        if not self._initialized or not (self.parallel_write or self.mpii.is_root):
            return  
        ##### prepare inputs
        t = kw.pop("t",step)
        dt = kw.pop("dt",1.0)
        dxyz = kw.pop("dxyz",1.0)
        to_center = kw.pop("to_center",False)
        # axis_labels = kw.pop("axis_labels",None)
        data_dict,ndim = self.prepare_data_dict(data,to_center)   
        local_offset = kw.pop("local_offset",[0]*ndim)
        global_offset = kw.pop("global_offset",[0.0]*ndim)
        if isinstance(dxyz, (int, float)):
            dxyz = [float(dxyz)] * ndim
            
        user_global_shape = kw.pop("global_shape",None)
        
        position = [0.5] if to_center else [0.0]

        
        ##### OpenPMD Iteration setup
        it = self.series.iterations[step]
        # it.open()     # use if iteration is expected to be closed
        it.set_time(t)
        it.set_dt(dt)
        it.set_time_unit_SI(1.0)
        mesh = it.meshes[name]
    
        #### setup mesh geometry
        if ndim == 2:
            local_offset = [local_offset[0], 0, local_offset[1]]
            global_offset = [global_offset[0], 0, global_offset[1]]
            dxyz = [dxyz[0], 1.0, dxyz[1]]
        else:
            raise ValueError(f"Unsupported data dimension: {ndim}")
    
        mesh.grid_spacing = dxyz
        mesh.grid_global_offset = global_offset
        mesh.data_order = 'C'
        mesh.axis_labels = ['x', 'y', 'z']
    
        #### write the data
        for comp_name, comp_data in data_dict.items():
            data_formatted = self._format_array_shape(comp_data)
            # Detect dataset shape PER COMPONENT
            if user_global_shape is not None:
                g_shape = [g - 1 for g in user_global_shape] if to_center else list(user_global_shape)
                comp_global_shape = [g_shape[0], 1, g_shape[1]] if ndim == 2 else g_shape
            else:
                # Auto-detect directly from this component's formatted array
                comp_global_shape = [int(s) for s in data_formatted.shape]
        
        
            component = mesh[comp_name]
            component.position = position * data_formatted.ndim
            component.reset_dataset(opmd.Dataset(data_formatted.dtype, comp_global_shape))
        
            local_slice = tuple(slice(off, off + size) for off, size in zip(local_offset, data_formatted.shape))
            component[local_slice] = data_formatted
    
        # it.close()   # dont close so that additional fields can be added to the iteration
        self.series.flush()
    
    def _format_array_shape(self, data):
        """Pads 2D arrays [Nz, Nx] into 3D [Nz, 1, Nx]."""
        return data.reshape(data.shape[0], 1, data.shape[1]) if data.ndim == 2 else data

    
    def prepare_data_dict(self,data,to_center=False):

        if isinstance(data, dict):
            data_dict = data
        else:
            data_dict = {opmd.Mesh.SCALAR: data}
        
        processed_dict = {}
        for comp, data in data_dict.items():
            arr = data.squeeze()
            if to_center:
                arr = self.to_cell_centered(arr,comp)
            arr = to_cpu(arr)
            processed_dict[comp] = arr
        
        return processed_dict, arr.ndim
    
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
        
    
    import numpy as np

    def to_cell_centered(self, data, comp=None, domain_cells=None):
        """
        GPU/CPU universal staggered-to-cell-center converter for a single array.
        Preserves device placement (CuPy, PyTorch, NumPy) with zero host transfers.
    
        Parameters
        ----------
        data : array-like (NumPy, CuPy, PyTorch, PyAMReX view)
            The raw field component array.
        comp : str, optional
            Component name ('x', 'y', 'z', 'Ex', 'Ez', 'rho', etc.).
        domain_cells : list/tuple of int, optional
            Domain cell counts [Nz, (Ny,) Nx]. If provided, shape-matching is used.
    
        Returns
        -------
        array-like
            The cell-centered array on the original GPU/CPU device.
        """
        arr = data.squeeze()
        nodal_axes = []
    
        # 1. Preferred method: Shape-matching against target domain cell counts
        if domain_cells is not None:
            for axis, (curr_len, target_len) in enumerate(zip(arr.shape, domain_cells)):
                if curr_len == target_len + 1:
                    nodal_axes.append(axis)
    
        # 2. Fallback: Deduce Yee grid nodal axes from component name in [z, (y,) x] ordering
        elif comp is not None:
            comp_key = str(comp).lower().replace("e", "").replace("b", "")
            ndim = arr.ndim
    
            if ndim == 2:  # Layout: [z, x]
                if comp_key == "x":
                    nodal_axes = [0]  # Ex is Nodal in z
                elif comp_key == "z":
                    nodal_axes = [1]  # Ez is Nodal in x
                elif comp_key in ("y", "rho", "phi", "node"):
                    nodal_axes = [0, 1]  # Nodal in both
            elif ndim == 3:  # Layout: [z, y, x]
                if comp_key == "x":
                    nodal_axes = [0, 1]  # Ex edge: Nodal in z, y
                elif comp_key == "y":
                    nodal_axes = [0, 2]  # Ey edge: Nodal in z, x
                elif comp_key == "z":
                    nodal_axes = [1, 2]  # Ez edge: Nodal in y, x
                elif comp_key in ("rho", "phi", "node"):
                    nodal_axes = [0, 1, 2]
    
        # 3. Perform GPU/CPU zero-copy slice averaging along nodal axes
        for axis in nodal_axes:
            slc_a = [slice(None)] * arr.ndim
            slc_b = [slice(None)] * arr.ndim
            slc_a[axis] = slice(0, -1)
            slc_b[axis] = slice(1, None)
    
            arr = 0.5 * (arr[tuple(slc_a)] + arr[tuple(slc_b)])
    
        return arr
    
    
    def close(self,):
        if self.parallel_write or self.mpii.is_root:
            self.series.flush()
            self.series.close()
     
        
def get_warpx_axis_labels(dims):
    dims = str(dims)  
    mapping = {
        "3": ["x", "y", "z"],
        "2": ["x", "y","z"],
        "RZ": ["r", "z"],
        "1": ["z"],
        }

    if dims in mapping:
        return mapping[dims]
    raise ValueError(f"Unsupported WarpX geometry dimension: {dims}")
    
def save_to_npy(field,path,suffix=None):
    global_data = field[...]   # this does an mpi gather on all ranks
    if mpit.get_rank() == 0:
        np.save(path, global_data)        

    
if __name__ == "__main__":
    pass
    

    
    