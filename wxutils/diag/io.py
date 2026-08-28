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




class IO():
    def __init__(self,**kw):
        self.path = Path(kw.pop("path","./diags"))
        
    def pre_initialize(self,sim):
        # callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installcallback("onbreaksignal",self.close)
        self.sim = sim
        self.author = getpass.getuser()
        self.software = "wxutils"
        self.version = version("wxutils")
        
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self.grid_info()
    
    
    def save(self,data):
        pass
    
    def close():
        pass
    
    
    
# class VizSchema1D(IO):
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         self.path = self.path /"history.h5"
#         self.datasets = []

#     def pre_initialize(self,sim):
#         super().pre_initialize(sim)
        
#         """Creates or resets the HDF5 file with VizSchema metadata and empty resizable datasets (Rank 0 only)."""
#         if mpit.get_rank() != 0:
#             return

#         with h5py.File(str(self.path), "w") as h5f:
#             mesh_group = h5f.create_group("runInfo")
#             mesh_group.attrs["software"] = np.bytes_(self.software)
#             mesh_group.attrs["version"] = np.bytes_(self.version)
            
#     def create_dataset(self,name,buf_size,dtype=np.float64):
#         if name in self.datasets:
#             raise Exception(f"'{name}' already in history dataset")
#         self.datasets.append(name)
#         with h5py.File(str(self.path), "w") as h5f:
#             timeGroupName = self.get_time_group_name(name)
#             mesh_group = h5f.create_group(timeGroupName)
#             mesh_group.attrs["vsType"] = np.bytes_("mesh")
#             mesh_group.attrs["vsKind"] = np.bytes_("rectilinear")
#             mesh_group.attrs["vsAxis0"] = np.bytes_("time")
            
#             mesh_group.create_dataset(
#                 "time", 
#                 shape=(0,), 
#                 maxshape=(None,), 
#                 chunks=(buf_size,), 
#                 dtype=dtype
#                 )
            
#             # Create empty resizable variable dataset
#             dset_var = h5f.create_dataset(
#                 name, 
#                 shape=(0,), 
#                 maxshape=(None,), 
#                 chunks=(buf_size,), 
#                 dtype=dtype
#                 )
#             dset_var.attrs["vsType"] = np.bytes_("variable")
#             dset_var.attrs["vsMesh"] = np.bytes_(timeGroupName)
    
#     def get_time_group_name(self,name):
#         return f"timeSeries{name}"
    
#     def save(self,name,data):
#         """Collective flush—reduces entire buffered vector across ranks at once."""
#         timeGroupName = self.get_time_group_name(name)
        
#         if mpit.get_rank() == 0:
#             n_samples = len(x0_local)
#             with h5py.File(str(self.path), "a") as h5f:
#                 dset_time = h5f[f"{timeGroupName}/time"]
#                 dset_var = h5f[self.name]
    
#                 old_size = dset_time.shape[0]
#                 new_size = old_size + n_samples
                
#                 dset_time.resize((new_size,))
#                 dset_var.resize((new_size,))
                
#                 dset_time[old_size:new_size] = x0_local
#                 dset_var[old_size:new_size] = x1_global
    
#         # 4. Reset local buffer index
#         self.buf_step = 0
        
#     def on_exit(self):
#         self.save()
    
    
        
class OpenPMD(IO):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.suffix_format = kw.pop("suffix_format","%06T")
        self.path = self.path /f"openpmd_{self.suffix_format}.bp5"
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        self.grid_info()
        self.series = opmd.Series(self.path, opmd.Access.create)
        self.series.author = self.author
        self.series.set_attribute("dependencies", f"{self.series.software} {self.series.software_version}")
        self.series.set_software(self.software,self.version)
        os.makedirs(self.path.parent,exist_ok=True)
        with open(self.path.parent/"paraview.pmd", "w") as f:
            f.write(f"openpmd_{self.suffix_format}.bp5\n")
    
    def post_initialize(self):
        super().post_initialize()
        
            
    def save(self,data):
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
        
        
def save_to_npy(field,path,suffix=None):
    global_data = field[...]   # this does an mpi gather on all ranks
    if mpit.get_rank() == 0:
        np.save(path, global_data)        

    
if __name__ == "__main__":
    pass
    

    
    