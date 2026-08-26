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
    rank = get_rank()
    
    # Early exit conditions where EVERY rank prints
    if (ranks == "all") or (isinstance(ranks, (int, float)) and ranks < 0):
        print(f"Rank {rank}: {string}", flush=True)
        return

    # Handle standard case (single rank or collection of ranks)
    allowed_ranks = ranks if isinstance(ranks, (list, tuple)) else [ranks]
    
    if rank in allowed_ranks:
        print(f"Rank {rank}: {string}", flush=True)
    