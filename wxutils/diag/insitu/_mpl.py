# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl

from pywarpx.LoadThirdParty import load_cupy
from pywarpx import particle_containers, picmi,callbacks

from wxutils.utils import to_cpu
import wxutils.mpitools as mpit

from types import MethodType

colors = 'brgk'

__all__ = ["InSituMPL"]

class InSituMPL():
    def __init__(self,backend='QtAgg',rate=1):
        self.backend=backend
        self.figs = {}
        self.axs = {}
        self.rate = rate
        mpl.use(self.backend)
        plt.ion()
    
    def pre_initialize(self,sim):
        if mpit.enabled():
            if mpit.get_rank() == 0:
                # raise Warning("matplotlib plotting is disabled with mpi runs")
                print("Warning: matplotlib plotting is disabled with mpi runs",flush=True )
            return
        self.sim = sim
        self.xp,_ = load_cupy()
        
    
    def add_fig(self,key,fig=1,figsize=(12,6)):
        if type(fig) is int: 
            fig = plt.figure(fig, figsize=figsize)
            ax = fig.subplots(nrows=1,ncols=1)
        else:
            ax = fig.axes
            ax = ax[0]
        self.figs[key], self.axs[key] = fig, ax
        # self.figs[key] = plt.figure(fig,figsize=figsize)
        # self.axs[key] = self.figs[key].subplots(nrows=1,ncols=1)
    
    def add_particles(self,particles):
        if not isinstance(particles,(str,list,tuple)):
            raise TypeError("particles must be of type string, list, or tuple")
        elif not isinstance(particles,(list,tuple)):
            self.particles = [particles]
        else:
            self.particles = particles
        callbacks.installcallback('particleinjection', lambda: self.plot_particles('particles'))
        
    def add_plot(self,func,cb_loc='particleinjection'):
        bound_method = MethodType(func, self)
        setattr(self,func.__name__,bound_method)
        callbacks.installcallback(cb_loc, getattr(self,func.__name__))
    
    
    def plot_particles(self,key):
        if self.sim.extension.warpx.getistep(0)% self.rate != 0:
            return 
        fig = self.figs[key]
        ax = self.axs[key]
        # print(ax)
        xymin = self.xp.array(self.sim.solver.grid.lower_bound)
        xymax = self.xp.array(self.sim.solver.grid.upper_bound)
        area = self.xp.prod(xymax - xymin)
        ax.clear() 
        ax.set(xlim=(xymin[0],xymax[0]),ylim=(xymin[1],xymax[1]))
        info = ''
        for i,particle in enumerate(self.particles):
            p_pc = self.sim.particles.get(particle)
            mpsum = 0.
            psum = 0.0
            for pti in p_pc.iterator(level=0):
                x = to_cpu(pti['x'])
                y = to_cpu(pti['z'])
                ax.scatter(x, y,c=colors[(i)%len(colors)],s=1)
    
            charge = p_pc.sum_particle_charge(0)
            mpsum = p_pc.total_number_of_particles()
            density = mpsum/area
            info += 'N%s = %i, RHO%s = %0.2e, '%(particle,mpsum,particle,density) 
        ax.set(title=info)
        #plot_impl(grid,implicit_expr,fig=1,clear=False)
        # plot_impl(grid,expression,fig=1,clear=False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # plt.gca().set_aspect('equal')
        
    def close(self):
        for fig in self.figs.values:
            if self.fig:
                self.fig.close()
            
            


