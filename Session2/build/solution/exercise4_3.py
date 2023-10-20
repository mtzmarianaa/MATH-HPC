# Exericise 4: All-to-all
# Source: https://subscription.packtpub.com/book/programming/9781785289583/3/ch03lvl1sec49/collective-communication-using-alltoall

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

senddata = rank*np.arange(size, dtype = int)

global_result1 = comm.reduce(senddata, op = MPI.SUM, root = 0)
global_result2 = comm.reduce(rank, op = MPI.MAX, root = 0)
global_result3 = comm.allreduce(senddata, op = MPI.SUM)
global_result4 = comm.allreduce(rank, op = MPI.MAX)

#Print
print("Process: ", rank,
      " operation1: ", global_result1,
      " operation2: ", global_result2,
      " operation3: ", global_result3,
      " operation4: ", global_result4)


