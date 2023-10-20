# Source: https://nyu-cds.github.io/python-mpi/05-collectives/
# (modified)
import numpy
from math import acos, cos, pi
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def integral(x_i, h, n):
    integ = 0.0
    for j in range(n):
        x_ij = x_i + (j + 0.5) * h
        integ += cos(x_ij) * h
    return integ

a = 0.0
b = pi / 2.0
my_int = 0
integral_sum = numpy.zeros(1)

# Initialize value of n only if this is rank 0
if rank == 0:
    n0 = 500 # default value n = 500
else:
    n0 = 0 # if not n = 0

# Broadcast n to all processes
n = comm.bcast(n0, root=0)

# Compute partition
h = (b - a) / (n * size) # calculate h *after* we receive n
x_i = a + rank * h * n
my_int = integral(x_i, h, n)

# Send partition back to root process, computing sum across all partitions
print("Process ", rank, " has the partial integral ", my_int)
integral_sum = comm.reduce(my_int, MPI.SUM, root = 0)

# Only print the result in process 0
if rank == 0:
    print('The Integral Sum =', integral_sum)
