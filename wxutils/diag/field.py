# -*- coding: utf-8 -*-
from .diag import DiagnosticBase
# import wxutils.mpitools as mpit
from wxutils.utils import to_cpu
import numpy as np
class Field(DiagnosticBase):
    def __init__(self, name, io,save_period,**kw):
        super().__init__(name, io=io,save_period=save_period,**kw)
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)

    def post_initialize(self):
        super().post_initialize()
        self.data = self.sim.fields.get(self.name,level=self.lev)
        
        
    def save(self):
        
        
        if self.io.parallel_write:
            ###### parallel write implementation
            raise NotImplementedError("Option 'parallel_write' is not implemented yet. It writes corrupted data.")
            reverse = False
            lev_info = self.get_lev_info(self.lev,reverse=reverse )
            global_info = self.get_global_info(reverse=reverse )
            ngrow = list(reversed(self.data.n_grow_vect)) if reverse else self.data.n_grow_vect
            
            for mfi in self.data:
                mfi_info = self.get_mfi_info(mfi,reverse=reverse)
                data = self.data.array(mfi).to_xp().squeeze()
                # data = data.T
                data = np.ascontiguousarray(to_cpu(data))
                data = self.trim_guard_cells(data,ngrow)
                self.io.save(
                    name=self.name,
                    data=data,
                    dxyz=lev_info.dxyz,
                    local_offset=mfi_info.local_offset,
                    global_offset=global_info.lower_bound,
                    global_shape=global_info.shape,
                    step=lev_info.step,
                    t=lev_info.t,
                    dt=lev_info.dt,
                    axis_labels=None,
                    node_to_center=True,
                    )
        else:
            ###### mpi gather implementation
            lev_info = self.get_lev_info(self.lev)
            global_info = self.get_global_info()
            data = self.data[...]
            if self.mpii.is_root:
                self.io.save(
                    name=self.name,
                    data=data,
                    dxyz=lev_info.dxyz,
                    local_offset=[0,0,0],
                    global_offset=global_info.lower_bound,
                    global_shape=global_info.shape,
                    step=lev_info.step,
                    t=lev_info.t,
                    dt=lev_info.dt,
                    axis_labels=None,
                    node_to_center=True,
                    )
    
        self.log_step = 0
    
    def log(self):
        """
        log just allows for storing data to memory and checks whether to save
        
        If only saved data is needed every save_period, then this should pass to save
        
        If some time filtering is needed, then this can be useful?
        """
        if (self.step % self.save_period == 0) or self._breaksignal:
            self.save()
        self.log_step += 1
    
    def trim_guard_cells(self, data,ngrow):
        """
        Squeezes 4D FAB array into 2D/3D and slices out guard cells to match valid domain.
        """
        if data.ndim == 2:
            # ngrow is [ng_x, ng_z]. In C-contiguous array [z, x], z is index 1, x is index 0
            ng_x, ng_z = ngrow[0], ngrow[1]
            slice_z = slice(ng_z, -ng_z) if ng_z > 0 else slice(None)
            slice_x = slice(ng_x, -ng_x) if ng_x > 0 else slice(None)
            return data[slice_x,slice_z]
    
        elif data.ndim == 3:
            ng_x, ng_y, ng_z = ngrow[0], ngrow[1], ngrow[2]
            slice_z = slice(ng_z, -ng_z) if ng_z > 0 else slice(None)
            slice_y = slice(ng_y, -ng_y) if ng_y > 0 else slice(None)
            slice_x = slice(ng_x, -ng_x) if ng_x > 0 else slice(None)
            return data[slice_x, slice_y, slice_z]
    
        return data