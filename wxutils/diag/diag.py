# -*- coding: utf-8 -*-
from wxutils.core.base import CallbackBase
from pywarpx import callbacks
import numpy as np

class DiagnosticBase(CallbackBase):
    def __init__(self, name,**kw):
        self.name = name
        self.io = kw.pop("io",None)
        self.save_period = int(kw.pop("save_period",2**32-1))
        self.callback_loc = kw.pop("callback_loc", "afterstep")
        self.log_period = int(kw.pop("log_period", 1))
        self.dtype = kw.pop("dtype",np.float64)
        self.dump_at_step_zero = kw.pop("dump_at_step_zero",True)
        self.log_step = 0
        self.lev = 0
        self.data_buffer_length = int(np.ceil(self.save_period/self.log_period))  # TODO, this needs to have a common denominator, no rounding
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        if self.dump_at_step_zero:
            callbacks.installcallback("afterInitEsolve",self._log)
        callbacks.installcallback(self.callback_loc,self._log)
        callbacks.installcallback("onbreaksignal",self._end)
        if self.io:
            self.io.initialize()
    
    def post_initialize(self):
        super().post_initialize()
        if self.io:
            self.io.create_dataset(
                name=self.name,
                buf_size=self.data_buffer_length,
                dtype=self.dtype
                )
        
    def _log(self):
        self.step = self.sim.extension.warpx.getistep(self.lev)
        if self.step % self.log_period == 0:
            self.log()
    
    def _end(self):
        self._breaksignal = True
        self.log()
        self._breaksignal = False  # incase sim.step() is run again
    
    def log(self):
        """
        log just allows for storing data to memory and checks whether to save
        
        If only saved data is needed every save_period, then this should pass to save
        
        If some time filtering is needed, then this can be useful?
        """
        # self.get()
        if (self.step % self.save_period == 0) or self._breaksignal:
            self.save()
        self.log_step += 1
        
    def save(self):
        if self.io:
            self.io.save(self.name,self.data)
        self.log_step = 0
    
class Diagnostic1D(DiagnosticBase):
    def __init__(self, name,io,save_period,**kw):
        super().__init__(name,io=io,save_period=save_period,**kw)
        self.log_step = 0
    
    def log(self):
        """Purely local logging—zero MPI latency or network communication per step."""
        x0,x1 = self.get()
        self.data[0, self.log_step] = x0
        self.data[1, self.log_step] = x1
        self.log_step += 1
        
        if self.log_step >= self.save_period:
            self.save()
    
    def get(self):
        """must return tuple of (time,value), this will get logged"""
        raise NotImplementedError("this must be defined by your diagnostic and return a tuple of (value,time)")
        
        
            
            
