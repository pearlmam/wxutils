# -*- coding: utf-8 -*-
import shutil

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

def get_rank():
    if mpi == None:
        return 0
    else:
        comm = mpi.COMM_WORLD
        return comm.Get_rank()
        
def delete_diagnostics():
    if mpi_enabled():
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    if rank == 0:
        shutil.rmtree('./diags',ignore_errors=True)
        
        
def calc_grid_blocking(nx,ny=None,nz=None):
    comm = mpi.COMM_WORLD
    nc = comm.Get_size()
    
    blocking_factor_x = 128
    max_grid_size_x=1024
    blocking_factor_z = 16
    max_grid_size_z=16
