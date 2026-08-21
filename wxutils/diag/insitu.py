# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl

from pywarpx.LoadThirdParty import load_cupy
from pywarpx import particle_containers, picmi,callbacks

from wxutils.callbacks.utils import to_cpu_array
from wxutils.utils import mpi_enabled
#from warpx_helpers.geoViz import plot_impl,gen_impl_vtk
from wxutils.utils import get_rank

colors = 'brgk'

class InSituMPL():
    def __init__(self,particles,backend='QtAgg',pausePeriod=None):
        self.backend=backend
        self.pausePeriod = pausePeriod
        if not isinstance(particles,(str,list,tuple)):
            raise TypeError("paritcles must be of type string, list, or tuple")
        elif not isinstance(particles,(list,tuple)):
            self.particles = [particles]
        else:
            self.particles = particles
        
        
    def pre_initialize(self,sim):
        if mpi_enabled():
            # raise Warning("matplotlib plotting is disabled with mpi runs")
            if get_rank() == 0:
                print("Warning: matplotlib plotting is disabled with mpi runs",flush=True )
            return
        
        self.sim = sim
        callbacks.installcallback('particleinjection', self.plot_particles)
        ### setup "in-situ" plotting
        self.xp,_ = load_cupy()
        mpl.use(self.backend)
        plt.ion()
        figsize = (12,6)
        self.fig = plt.figure(1,figsize=figsize)
        self.ax = self.fig.subplots(nrows=1,ncols=1)

    def plot_particles(self):
        
        xymin = self.xp.array(self.sim.solver.grid.lower_bound)
        xymax = self.xp.array(self.sim.solver.grid.upper_bound)
        area = self.xp.prod(xymax - xymin)
        self.ax.clear() 
        self.ax.set(xlim=(xymin[0],xymax[0]),ylim=(xymin[1],xymax[1]))
        info = ''
        for i,particle in enumerate(self.particles):
            p_pc = self.sim.particles.get(particle)
            mpsum = 0.
            psum = 0.0
            for pti in p_pc.iterator(level=0):
                x = to_cpu_array(pti['x'])
                y = to_cpu_array(pti['z'])
                self.ax.scatter(x, y,c=colors[(i)%len(colors)],s=1)
    
            charge = p_pc.sum_particle_charge(0)
            mpsum = p_pc.total_number_of_particles()
            density = mpsum/area
            info += 'N%s = %i, RHO%s = %0.2e, '%(particle,mpsum,particle,density) 
        self.ax.set(title=info)
        #plot_impl(grid,implicit_expr,fig=1,clear=False)
        # plot_impl(grid,expression,fig=1,clear=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.gca().set_aspect('equal')