# -*- coding: utf-8 -*-
import inspect
import numpy as np
from pywarpx import callbacks
from pywarpx.LoadThirdParty import load_cupy
from wxutils.utils import get_warpx_axis_labels
import wxutils.mpitools as mpit

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

class GridInfo():
    def __init__(self,sim,xp=np):
        self.dims = sim.solver.grid.number_of_dimensions
        self.number_of_cells = sim.solver.grid.number_of_cells
        self.lower_bound = sim.solver.grid.lower_bound
        self.upper_bound = sim.solver.grid.upper_bound
        self.dxyz = xp.array((xp.array(self.upper_bound)-xp.array(self.lower_bound))/xp.array(self.number_of_cells))  # should this us xp?
        self.axis_labels = get_warpx_axis_labels(self.dims)


class CallbackBase(Setup):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def pre_initialize(self,sim):
        self.sim = sim
        self._breaksignal = False
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        
    def post_initialize(self,):
        self.xp, _ = load_cupy()
        self.mpii = mpit.Info()

    def concat(self, list_of_arrays,*args,**kwargs):
        if len(list_of_arrays) == 0:
            return self.xp.empty(0)
        else:
            return self.xp.concatenate(list_of_arrays,*args,**kwargs)

if __name__ == "__main__":
    class Test(CallbackBase):
        def __init__(self,species,sigma,name="theName"):
            super().__init__(species,sigma,name)
        
    test = Test("electrons",5.0, name="other name")