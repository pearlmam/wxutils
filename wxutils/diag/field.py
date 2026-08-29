# -*- coding: utf-8 -*-
from .diag import DiagnosticBase
from wxutils.core.base import GridInfo

class Field(DiagnosticBase):
    def __init__(self, name, io,save_period,**kw):
        super().__init__(name, io,save_period,**kw)
        
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        self.grid_info = GridInfo(sim)

    def post_initialize(self):
        super().post_initialize()
        self.data = self.sim.fields.get(self.name,level=self.lev)
        
        
    def save(self):
        # I probably need to pass all data from here.
        data = self.data[...]
        self.io.save(self.name,data,grid=self.grid_info)
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