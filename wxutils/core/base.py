# -*- coding: utf-8 -*-
import inspect
from pywarpx import callbacks
from pywarpx.LoadThirdParty import load_cupy
import wxutils.mpitools as mpit
from typing import NamedTuple

class Setup:
    def __init__(self, *args, **kwargs):
        # 1. Get the signature of the final child class (e.g., Test)
        # We look up the __init__ method of whatever self actually is
        sig = inspect.signature(self.__init__)
        
        # 2. Bind the passed *args and **kwargs to the parameters defined in that signature
        bound_args = sig.bind(*args, **kwargs)
        
        # 3. Apply default values for any arguments that weren't explicitly passed
        bound_args.apply_defaults()
        
        # 4. Automatically set everything as instance variables
        for key, value in bound_args.arguments.items():
            if key != 'self':  # Skip the self reference itself
                setattr(self, key, value)

# class GridInfo():
#     def __init__(self,sim,xp=np):
#         self.dims = sim.solver.grid.number_of_dimensions
#         self.number_of_cells = sim.solver.grid.number_of_cells
#         self.lower_bound = sim.solver.grid.lower_bound
#         self.upper_bound = sim.solver.grid.upper_bound
#         self.dxyz = xp.array((xp.array(self.upper_bound)-xp.array(self.lower_bound))/xp.array(self.number_of_cells))  # should this us xp?
#         self.axis_labels = get_warpx_axis_labels(self.dims)
        

class CallbackBase(Setup):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._initialized = False
        
    def pre_initialize(self,sim):
        self.sim = sim
        self._breaksignal = False
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        
    def post_initialize(self,):
        self.xp, _ = load_cupy()
        self.mpii = mpit.Info()
        self._initialized = True

    def concat(self, list_of_arrays,*args,**kwargs):
        if len(list_of_arrays) == 0:
            return self.xp.empty(0)
        else:
            return self.xp.concatenate(list_of_arrays,*args,**kwargs)
        
    def get_mfi_info(self, mfi,reverse=False):
        fab_box = mfi.fabbox()
        valid_box = mfi.validbox()
        local_offset_lo =  valid_box.small_end
        local_offset_hi = valid_box.big_end
        fab_offset_lo = fab_box.small_end
        fab_offset_hi = fab_box.big_end
        ng_lo = local_offset_lo - fab_offset_lo
        ng_hi = fab_offset_hi - local_offset_hi
        if reverse:
            local_offset_lo = reversed(valid_box.small_end)
            local_offset_hi = reversed(valid_box.big_end)
            fab_offset_lo = reversed(fab_box.small_end)
            fab_offset_hi = reversed(fab_box.big_end)
            ng_lo = reversed(ng_lo)
            ng_hi = reversed(ng_hi)
            
        return MFIInfo(
            local_offset=list(local_offset_lo),
            fab_offset=list(fab_offset_lo),
            ng_lo=list(ng_lo),                              # Guard cells at lower boundary [x, z]
            ng_hi=list(ng_hi)
            )
    
    def get_lev_info(self,lev=0,reverse=False):
        geom = self.sim.extension.warpx.Geom(lev)
        dxyz = geom.data().dx if not reverse else list(reversed(geom.data().dx))
        return LevelInfo(
            lev=lev,
            dxyz=dxyz,
            step=self.sim.extension.warpx.getistep(lev),
            t=self.sim.extension.warpx.gett_new(lev),
            dt=self.sim.extension.warpx.getdt(lev)
            )
    
    def get_global_info(self,reverse=False):
        lower_bound=self.sim.solver.grid.lower_bound
        upper_bound=self.sim.solver.grid.upper_bound
        number_of_cells=self.sim.solver.grid.number_of_cells
        if reverse:
            lower_bound=list(reversed(lower_bound))
            upper_bound=list(reversed(upper_bound))
            number_of_cells=list(reversed(number_of_cells))

        return GlobalInfo(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            number_of_cells=number_of_cells,
            shape=[x + 1 for x in number_of_cells]
            )
        

class LevelInfo(NamedTuple):
    lev: int
    dxyz: list[float]
    step: int
    t: float
    dt: float
    
class GlobalInfo(NamedTuple):
    lower_bound: list[float]
    upper_bound: list[float]
    number_of_cells: list[int]
    shape: list[int]
    
class MFIInfo(NamedTuple):
    local_offset: list[int]
    fab_offset: list[int]
    ng_lo: list[int]
    ng_hi: list[int]



if __name__ == "__main__":
    class Test(CallbackBase):
        def __init__(self,species,sigma,name="theName"):
            super().__init__(species,sigma,name)
        
    test = Test("electrons",5.0, name="other name")