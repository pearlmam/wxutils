# -*- coding: utf-8 -*-

import ascent
import conduit
import conduit.blueprint
from pywarpx import warpx, particle_containers
from pywarpx.LoadThirdParty import load_cupy
from pywarpx import particle_containers, picmi,callbacks
from wxutils.utils import to_cpu

colors = 'brgk'
class InSituAscent():
    def __init__(self,particles):
        # Initialize Ascent engine
        if not isinstance(particles,(str,list,tuple)):
            raise TypeError("paritcles must be of type string, list, or tuple")
        elif not isinstance(particles,(list,tuple)):
            self.particles = [particles]
        else:
            self.particles = particles
        
        self.a = ascent.Ascent()
        self.ascent_opts = conduit.Node()
        self.a.open(self.ascent_opts)
        
    def pre_initialize(self,sim):
        self.sim = sim
        callbacks.installcallback('particleinjection', self.plot_particles)
        ### setup "in-situ" plotting
        self.xp,_ = load_cupy()
        
    def post_initialize(self):
        
        for i,particle in enumerate(self.particles):
            p_pc = self.sim.particles.get(particle)

            for pti in p_pc.iterator(level=0):
                x = to_cpu(pti['x'])
                y = to_cpu(pti['z'])

        bp = conduit.Node()
        
        
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
                x = to_cpu(pti['x'])
                y = to_cpu(pti['z'])
                self.ax.scatter(x, y,c=colors[(i)%len(colors)],s=1)
    
            charge = p_pc.sum_particle_charge(0)
            mpsum = p_pc.total_number_of_particles()
            density = mpsum/area
            info += 'N%s = %i, RHO%s = %0.2e, '%(particle,mpsum,particle,density) 
        
        