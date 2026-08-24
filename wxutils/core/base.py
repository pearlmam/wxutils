# -*- coding: utf-8 -*-
import inspect
import numpy as np
from pywarpx import callbacks
from pywarpx.LoadThirdParty import load_cupy

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



class CallbackBase(Setup):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def pre_initialize(self,sim):
        self.sim = sim
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        
    def post_initialize(self,):
        self.xp, _ = load_cupy()
        
    def grid_info(self):
        if not hasattr(self,'sim'):
            raise AttributeError(f"class {self.__class__.__name__} must be run pre_initialize() before grid_info() can be called.")
        self.dims = self.sim.solver.grid.number_of_dimensions
        self.number_of_cells = self.sim.solver.grid.number_of_cells
        self.lower_bound = self.sim.solver.grid.lower_bound
        self.upper_bound = self.sim.solver.grid.upper_bound
        self.dxyz = self.xp.array((self.xp.array(self.upper_bound)-self.xp.array(self.lower_bound))/self.xp.array(self.number_of_cells))
        
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