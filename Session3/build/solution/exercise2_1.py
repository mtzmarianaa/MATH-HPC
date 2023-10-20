from mpi4py import MPI
import numpy as np

# Testing what comm.Split() does

# Initialize MPI (world)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Defining the subset assignment
if rank%2 == 0:
    color1 = 0
else:
    color1 = 1

if int(rank/2) == 0:
    color2 = 0
else:
    color2 = 1

new_comm1 = comm.Split(color = color1, key = rank)
new_rank1 = new_comm1.Get_rank()
new_size1 = new_comm1.Get_size()

new_comm2 = comm.Split(color = color2, key = rank)
new_rank2 = new_comm2.Get_rank()
new_size2 = new_comm2.Get_size()

print("Original rank: ", rank,
      " color1: ", color1,
      " new rank1: ", new_rank1,
      " color2: ", color2,
      " new rank2: ", new_rank2)
  
