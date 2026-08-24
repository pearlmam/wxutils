# -*- coding: utf-8 -*-

try:
    from mpi4py import MPI as mpi
except ImportError:
    mpi = None
  
def enabled():
    if mpi is None:
        return False
    else:
        comm = mpi.COMM_WORLD
        return comm.Get_size() > 1

def get_rank():
    if mpi is None:
        return 0
    else:
        comm = mpi.COMM_WORLD
        return comm.Get_rank()

def get_size():
    if mpi is None:
        return None
    else:
        comm = mpi.COMM_WORLD
        return comm.Get_size()

def get_comm():
    if mpi is None:
        return None
    else:
        return mpi.COMM_WORLD

def mpi_print(string, ranks=0):
    flush = True
    if not isinstance(ranks,(list,tuple,str)):
        ranks = [ranks]
    elif isinstance(ranks,(str)) and ranks.lower() == "all":
        print(f"Rank {get_rank()}: {string}", flush=flush)
        return
    if get_rank() in ranks:
        print(f"Rank {get_rank()}: {string}", flush=flush)
    