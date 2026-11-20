# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backend_bases import NavigationToolbar2

from pywarpx.LoadThirdParty import load_cupy
from pywarpx import particle_containers, picmi,callbacks

from wxutils.core.base import CallbackBase
from wxutils.utils import to_cpu
import wxutils.mpitools as mpit
from wxutils.features.helpers import set_species_params

_MISSING = object()
from ._helpers import update_home_decorator, move_figure

colors = 'brgk'

__all__ = ["InSituMPL","InSituParticles","InSituFigure"]

class InSituMPL(CallbackBase):
    def __init__(self,backend='QtAgg',**kw):
        self.backend=backend
        self.rate = kw.pop("rate",1)
        self.Figs = []
        self.syncVars = kw.pop("syncVars",[])
        if mpit.get_rank() == 0:
            mpl.use(self.backend)
            plt.ion()
    
    def pre_initialize(self,sim):
        super().pre_initialize(sim)
        self.mpii = mpit.Info()
        if self.syncVars:
            callbacks.installcallback("afterstep", self._sync_callback)
        self._sync_callback()
        self.time = []
        for fig in self.Figs:
            fig.pre_initialize(sim)
              
    def add_fig(self,fig):
        fig.parent = self
        self.Figs.append(fig)
    
    def set_var(self, name, value,setFigs=False):
        if hasattr(self, name):
            setattr(self, name, value)
        if setFigs:
            for fig in self.Figs:
                if hasattr(fig, name):
                    setattr(fig, name, value)
    
    def update(self):
        for fig in self.Figs:
            fig.update_plot()
    
    def _sync_callback(self):
        comm = getattr(self.mpii, 'comm')
        rank = self.mpii.rank

        # 1. Rank 0 flushes GUI events so pending widget clicks update attributes BEFORE broadcasting
        if rank == 0:
            for fig in self.Figs:
                if getattr(fig, "fig", None) is not None:
                    fig.fig.canvas.flush_events()

            # Collect current values from Rank 0 (falls back to fig attribute if not on self)
            payload = {}
            for var in self.syncVars:
                if hasattr(self, var):
                    payload[var] = getattr(self, var)
                elif self.Figs and hasattr(self.Figs[0], var):
                    payload[var] = getattr(self.Figs[0], var)
        else:
            payload = None

        # 2. Single collective broadcast for all synced variables at once
        payload = comm.bcast(payload, root=0)

        # 3. Apply broadcasted values to InSituMPL and every figure on ALL ranks
        for var, val in payload.items():
            if hasattr(self, var):
                setattr(self, var, val)
            for fig in self.Figs:
                if hasattr(fig, var):
                    setattr(fig, var, val)

    def update_time_vector(self,factor=1.):
        time = self.sim.extension.warpx.gett_new(0)*factor
        if len(self.time) == 0:
            self.time.append(time)
        elif time != self.time[-1]:
            self.time.append(time)
        else:
            pass
        return self.time
    
    def save(self):
        for fig in self.Figs:
            fig.save()
    
    def close(self):
        if mpit.get_rank() == 0:
            for fig in self.Figs.values:
                if self.fig:
                    self.fig.close()



class InSituFigure(CallbackBase):
    """future interactive figure"""
    def __init__(self,**kw):
        self.rate = kw.pop("rate",1)
        self.callback_loc = kw.pop("callback_loc", "afterstep")
        self.custom_home = kw.pop("custom_home",None)
        self.io = kw.pop("io",None)
        self.name = kw.pop("name",None)
        self.kwfig = kw
        
    def pre_initialize(self,sim):
        self.sim = sim
        self.mpii = mpit.Info()
        self.figure(**self.kwfig)
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installcallback(self.callback_loc, self.update_figure)
        if self.io:
            # This doesnt work, need steps end callback loc
            callbacks.installcallback("onbreaksignal",self.save)
    
    def post_initialize(self,):
        self.xp, _ = load_cupy()
        # if self.mpii.rank == 0:
        #     self.ax.clear() 

    def figure(self,fig=None,**kw):
        if self.mpii.rank != 0:
            self.fig = None
            self.ax = None
            return
        
        if isinstance(fig, (mpl.figure.Figure)):
            self.fig = fig
        else:
            nrows = kw.pop("nrows",1)
            ncols = kw.pop("ncols",1)
            position = kw.pop("position",None)
            self.fig = plt.figure(**kw)
            self.ax = self.fig.subplots(nrows=nrows,ncols=ncols)
            if position:
                self.move_figure(position[0], position[1])
        if self.custom_home:
            toolbar = self.fig.canvas.toolbar
            if toolbar is not None:
                toolbar._update_view = update_home_decorator(toolbar._update_view)

    def update_data(self):
        return self.sim.extension.warpx.getistep(0)% self.rate == 0
    
    def update_figure(self):
        return self.sim.extension.warpx.getistep(0)% self.rate == 0 and self.mpii.rank == 0
            
    def move_figure(self,pos):
        move_figure(self.fig, pos)
    
    def update_time_vector(self,factor=1.):
        time = self.sim.extension.warpx.gett_new(0)*factor
        if len(self.time) == 0:
            self.time.append(time)
        elif time != self.time[-1]:
            self.time.append(time)
        else:
            pass
        return self.time
    
    def save(self):
        if self.io:
            self.io.save(self.name,self.data)
    
    def close(self):
        self.fig.close()
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        parent = getattr(self, "parent", None)
        if (
            parent is not None
            and getattr(parent, "mpii", None) is not None
            and parent.mpii.rank == 0
            and name in getattr(parent, "syncVars", ())
            and getattr(parent, name, _MISSING) != value
        ):
            parent.set_var(name, value)

    
class InSituParticles(InSituFigure):
    def __init__(self,particles,**kw):
        self.scatter_kw = kw.pop("scatter_kw",{})
        super().__init__(**kw)
        self.add_particles(particles)
        self.name="particles"
        
    
    def add_particles(self,particles):
        if not isinstance(particles,(list,tuple)):
            particles = [particles]
        else:
            particles = particles
        
        self.particles = {}
        for particle in particles:
            self.particles[particle.name] = set_species_params(particle)
        
    def post_initialize(self,):
        super().post_initialize()
        if self.mpii.rank == 0:
            xymin = self.xp.array(self.sim.solver.grid.lower_bound)
            xymax = self.xp.array(self.sim.solver.grid.upper_bound)
            # self.ax.clear() 
            self.ax.set(xlim=(xymin[0],xymax[0]),ylim=(xymin[1],xymax[1]))
        
        self._initialized = True
    
    def update_figure(self):
        info = ''
        for i,particle in enumerate(self.particles.keys()):
            p_pc = self.sim.particles.get(particle)
            mpsum = 0.
            x_local_list = []
            y_local_list = []
            for pti in p_pc.iterator(level=0):
                x_local_list.append(to_cpu(pti["x"]))
                y_local_list.append(to_cpu(pti["z"]))
    
            if len(x_local_list) > 0:
                x_local = self.xp.concatenate(x_local_list)
                y_local = self.xp.concatenate(y_local_list)
            else:
                x_local = self.xp.empty(0, dtype=self.xp.float64)
                y_local = self.xp.empty(0, dtype=self.xp.float64)
            
            x_gathered = self.mpii.comm.gather(x_local, root=0)
            y_gathered = self.mpii.comm.gather(y_local, root=0)    
            
            if self.mpii.rank == 0:
                # Filter out empty arrays before concatenating
                x_non_empty = [arr for arr in x_gathered if len(arr) > 0]
                y_non_empty = [arr for arr in y_gathered if len(arr) > 0]

                scatter_kw = self.scatter_kw.get(particle,{"s":1})
                if len(x_non_empty) > 0:
                    x_global = self.xp.concatenate(x_non_empty)
                    y_global = self.xp.concatenate(y_non_empty)
                else:
                    x_global = self.xp.empty(0, dtype=self.xp.float64)
                    y_global = self.xp.empty(0, dtype=self.xp.float64)
            
                if self.sim.extension.warpx.getistep(0) <= 1:
                    self.ax.scatter(x_global, y_global,c=colors[(i)%len(colors)],**scatter_kw)
                else:
                    scat = self.ax.collections[i]
                    scat.set_offsets(self.xp.column_stack((x_global, y_global)))  # or np.c_[new_x, new_y]
                    
                mpsum = len(x_global)
                # density = mpsum / area if area > 0 else 0.0
                info += f"N{particle} = {mpsum},"
        return self.sim.extension.warpx.getistep(0)% self.rate == 0
                
        if self.mpii.rank == 0:
            t = self.sim.extension.warpx.gett_new(0)*1e9
            info += f" time = {t:.2f} ns"
            #self.ax.set(title=info)
            #plot_impl(grid,implicit_expr,fig=1,clear=False)
            # plot_impl(grid,expression,fig=1,clear=False)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            # plt.gca().set_aspect('equal')





