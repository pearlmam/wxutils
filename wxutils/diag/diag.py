# -*- coding: utf-8 -*-
from wxutils.base import CallbackBase
from pywarpx import callbacks

class DiagnosticBase(CallbackBase):
    def __init__(self, name,io,save_period,**kw):
        self.name = name
        self.io = io
        self.save_period = save_period
        self.callback_loc = kw.pop("callback_loc", "afterstep")
        self.log_period = int(kw.pop("log_period", 1))
        
        
        self.lev = 1
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        callbacks.installcallback(self.callback_loc,self._log)
        callbacks.installcallback("onbreaksignal",self._end)
        self.io.pre_initialize(sim)
        
    def post_initialize(self):
        super().post_initialize()
        
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
        if self.buf_step >= self.save_period or self._breaksignal:
            self.save()
    
    def save(self):
        self.io.save(self.name,self.data)
    
class Diagnostic1D(DiagnosticBase):
    def __init__(self, name,io,save_period,**kw):
        super().__init__(name,io,save_period,**kw)
        self.buf_step = 0
    
    def log(self):
        """Purely local logging—zero MPI latency or network communication per step."""
        x0,x1 = self.get()
        self.data[0, self.buf_step] = x0
        self.data[1, self.buf_step] = x1
        self.buf_step += 1
        
        if self.buf_step >= self.buf_size:
            self.save()
    
    def get(self):
        """must return tuple of (value,time), this will get logged"""
        raise NotImplementedError("this must be defined by your diagnostic and return a tuple of (value,time)")
        
        
            
            
