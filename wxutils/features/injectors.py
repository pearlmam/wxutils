# -*- coding: utf-8 -*-


from pywarpx.callbacks import installcallback
from pywarpx import picmi
import numpy as np

def dimension_string_to_num(dimString,dim=3):
    if dim in [0,1]:
        if dimString == 'x':
            return 0
        elif dimString == 'z':
            return 1
    if dim in [2]:
        if dimString == 'x':
            return 0
        elif dimString == 'y':
            return 1
        elif dimString == 'z':
            return 2

class RealtimeFluxDistribution():
    """
    Describes a flux of particles emitted from a plane

    Parameters
    ----------
    flux: string
        Analytic expression describing flux of particles [m^-2.s^-1]
        Expression should be in terms of the position and time, written as 'x', 'y', 'z', and 't'.

    flux_normal_axis: string
        x, y, or z for 3D, x or z for 2D, or r, t, or z in RZ geometry

    surface_flux_position: double
        location of the injection plane [m] along the direction
        specified by `flux_normal_axis`

    flux_direction: int
        Direction of the flux relative to the plane: -1 or +1

    lower_bound: vector of floats, optional
        Lower bound of the distribution [m]

    upper_bound: vector of floats, optional
        Upper bound of the distribution [m]

    rms_velocity: vector of floats, default=[0.,0.,0.]
        Thermal velocity spread [m/s]

    directed_velocity: vector of floats, default=[0.,0.,0.]
        Directed, average, proper velocity [m/s]

    flux_tmin: float, optional
        Time at which the flux injection will be turned on.

    flux_tmax: float, optional
        Time at which the flux injection will be turned off.

    gaussian_flux_momentum_distribution: bool, optional
        If True, the momentum distribution is v*Gaussian,
        in the direction normal to the plane. Otherwise,
        the momentum distribution is simply Gaussian.
    """

    def __init__(self, sim,flux, flux_normal_axis,
                 surface_flux_position, flux_direction,
                 lower_bound = [None,None,None],
                 upper_bound = [None,None,None],
                 rms_velocity = [0.,0.,0.],
                 directed_velocity = [0.,0.,0.],
                 flux_tmin = None,
                 flux_tmax = None,
                 gaussian_flux_momentum_distribution = None,
                 **kw):
        self.sim = sim
        self.flux = flux
        self.flux_normal_axis = dimension_string_to_num(flux_normal_axis,sim.solver.grid.number_of_dimensions)
        self.surface_flux_position = surface_flux_position
        self.flux_direction = flux_direction
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rms_velocity = rms_velocity
        self.directed_velocity = directed_velocity
        self.flux_tmin = flux_tmin
        self.flux_tmax = flux_tmax
        self.gaussian_flux_momentum_distribution = gaussian_flux_momentum_distribution
        
        self.dt = sim.time_step_size
        self.number_per_cell_each_dim = np.array([5,5])
        self.dims = sim.solver.grid.number_of_dimensions
        self.cells = sim.solver.grid.number_of_cells
        self.lower_bound = sim.solver.grid.lower_bound
        self.upper_bound = sim.solver.grid.upper_bound
        self.emission_area = self.calc_emission_area(self.lower_bound,self.upper_bound)
        self.cells_on_plane = self.calc_cells_on_plane(self.cells)
        self.particles_per_cell = self.calc_particles_per_cell(self.number_per_cell_each_dim)
        self.dxyz = (np.array(self.upper_bound)-np.array(self.lower_bound))/np.array(self.cells)
        
        self.N = self.calc_number_emission_particles(self.directed_velocity)
        self.thickness = self.calc_emission_extents(self.directed_velocity)
        
        
    def inject(self,particle,flux=None,velocity=None):
        if flux is None:
            flux = self.flux
        if velocity is None:
            velocity = self.directed_velocity
        # more checks ??
        
        #### calculate particle positions.
        if (self.flux>0.0):
            N = self.calc_number_emission_particles(velocity)
            extents = self.calc_emission_extents(velocity)
            pos = [None]*3
            vel = [None]*3
            for dim in range(self.dims):
                pos[dim] = np.random.uniform(low=extents[dim][0],high=extents[dim][1], size=N)
                vel[dim] = velocity[dim] + np.random.normal(0, self.rms_velocity[0], size=N)
            w = self.calc_weight(flux,N)
            pc = self.sim.particles.get(particle)   # particle container
            pc.add_particles(x=pos[0],y=pos[2],z=pos[1],ux=vel[0],uy=vel[2],uz=vel[1],w=w)
        
    def calc_number_emission_particles(self,velocity):
        N = np.zeros(3)
        for dim in range(self.dims):
            if dim == self.flux_normal_axis:
                N[dim] = (self.number_per_cell_each_dim[dim]*velocity[dim]/self.dxyz[dim]*self.dt)
            else:
                N[dim] = (self.number_per_cell_each_dim[dim]*self.cells[dim])
        N = N.sum()
        # N = self.particles_per_cell*self.cells_on_plane/self.dxyz[self.flux_normal_axis]*self.dt*velocity[self.flux_normal_axis]
        n_int = int(N)
        fraction = N - n_int
        add_extra = 1 if np.random.random() < fraction else 0
        return n_int + add_extra

    
    
    def calc_emission_extents(self,velocity):
        extents = [[0.0,1.0]]*3
        for dim in range(self.dims):
            if dim == self.flux_normal_axis:
                extents[dim] = [self.surface_flux_position, self.surface_flux_position + velocity[dim]*self.dt*self.flux_direction]
            else:
                extents[dim] = [self.lower_bound[dim],self.upper_bound[dim]]
        return extents
    
    def calc_weight(self,flux,N):
        # area = np.prod(np.diff(np.array(extents),axis=1))
        # return flux/(velocity[self.flux_normal_axis])*area/N
        return flux*self.emission_area/N*self.dt
    
    def calc_emission_area(self,lower_bound,upper_bound):
        domain_lengths = np.diff(np.array([lower_bound,upper_bound]),axis=0)
        domain_lengths = np.delete(domain_lengths,(self.flux_normal_axis))
        return np.prod(domain_lengths)
    
    def calc_cells_on_plane(self,cells):
        cells = np.delete(cells,self.flux_normal_axis)
        return np.prod(cells)
    
    def calc_particles_per_cell(self,number_per_cell_each_dim):
        number_per_cell_each_dim = np.delete(self.number_per_cell_each_dim,self.flux_normal_axis)
        return  np.prod(number_per_cell_each_dim)
        
        