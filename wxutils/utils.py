# -*- coding: utf-8 -*-

try:
    from mpi4py import MPI as mpi
except ImportError:
    mpi = None
  
def mpi_enabled():
    if mpi == None:
        return False
    else:
        comm = mpi.COMM_WORLD
        return comm.Get_size() > 1
    
def calc_grid_blocking(nx,ny=None,nz=None):
    comm = mpi.COMM_WORLD
    nc = comm.Get_size()
    
    blocking_factor_x = 128
    max_grid_size_x=1024
    blocking_factor_z = 16
    max_grid_size_z=16
