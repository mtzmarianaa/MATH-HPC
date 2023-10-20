# Exercise 2, sending messages to processes
# sources:
#https://mpi4py.readthedocs.io/en/stable/tutorial.html#point-to-point-communication
#https://nyu-cds.github.io/python-mpi/03-nonblocking/

from mpi4py import MPI
import numpy as np



# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    req = comm.isend(data, dest=1, tag=11)
    req.wait()
    print("From process: ", rank, "\n data sent:", data, "\n")
elif rank == 1:
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
    print("From process: ", rank, "\n data received:", data, "\n")
elif rank == 2:
    data = np.array([1, 1, 1, 1, 1])
    req = comm.isend(data, dest=3, tag = 66)
    print("From process: ", rank, "\n data sent:", data, "\n")
else:
    req = comm.irecv(source = 2, tag = 66)
    data = req.wait()
    print("From process: ", rank, "\n data received:", data, "\n")








