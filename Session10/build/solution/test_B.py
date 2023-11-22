import numpy as np
from numpy.linalg import norm, cholesky, solve, qr, svd, lstsq
from pandas import read_csv
from numpy.random import normal
from math import ceil, log, sqrt, floor, log2
import matplotlib.pyplot as plt
import time
from random import sample
import random
import torch
from hadamard_transform import hadamard_transform
from mpi4py import MPI

# My classes
from create_matrix import create_matrix
from sketching_matrices import sketching_matrices
from TSQR import TSQR

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 2 ** 11

A = None
Omega1 = None
Omega1T = None
n_column = 0
if rank % int(sqrt(size)) == 0:
    R = [5, 10, 20]
    epsilon = 10 ** (-2)
    fun = create_matrix(n, R[0])
    A = fun.psd_Noise(epsilon)
    # print("A: ", A)

    sketch = sketching_matrices()
    Omega1 = sketch.SHRT_serial(n, n, epsilon=5, seed=123445)
    n_column = Omega1.shape[1]
    arrs = np.split(Omega1, n, axis=0)
    raveled = [np.ravel(arr) for arr in arrs]
    Omega1T = np.concatenate(raveled)

n_column = comm.bcast(n_column, root=0)

if rank < int(sqrt(size)):
    color_col = 0
    key_raw = 0
elif int(sqrt(size)) <= rank < 2 * int(sqrt(size)):
    color_col = 1
    key_raw = 1
else:
    color_col = 2
    key_raw = 2

if rank % int(sqrt(size)) == 0:
    color_raw = 0
    key_col = 0
elif (rank-1) % int(sqrt(size)) == 0:
    color_raw = 1
    key_col = 1
else:
    color_raw = 2
    key_col = 2

comm_raw = comm.Split(color=color_col, key=key_col)
rank_raw = comm_raw.Get_rank()
size_raw = comm_raw.Get_size()

comm_col = comm.Split(color=color_raw, key=key_raw)
rank_col = comm_col.Get_rank()

# Select raw
n_blocks = int(n/int(sqrt(size)))
submatrix = np.empty((n_blocks, n), dtype = 'd')
comm_raw.Scatterv(A, submatrix, root=0)

# Then we scatter the column
blockMatrix = np.empty((n_blocks, n_blocks), dtype = 'd')
receiveMat = np.empty((n_blocks*n_blocks), dtype = 'd')
arrs = np.split(submatrix, n, axis=1)
raveled = [np.ravel(arr) for arr in arrs]
submatrixT = np.concatenate(raveled)
comm_col.Scatterv(submatrixT, receiveMat, root = 0)
subArrs = np.split(receiveMat, n_blocks)
raveled2 = np.array([np.ravel(arr, order='F') for arr in subArrs])
blockMatrix = raveled2.reshape((n_blocks, n_blocks))

# print("rank_raw: ", rank_raw, "blockMatrix", blockMatrix)

# Scatter Omega1 along rows
Omega1_local = np.empty((n_blocks, n_column), dtype='d')
receiveOmega = np.empty((n_blocks*n_column), dtype = 'd')
comm_raw.Scatterv(Omega1T, receiveOmega, root=0)
subArrs = np.split(receiveOmega, n_column)
raveled2 = np.array([np.ravel(arr, order='F') for arr in subArrs])
Omega1_local = raveled2.reshape((n_blocks, n_column))

# print("rank_raw: ", rank, "Omega1_local: ", Omega1_local)


local_mult = np.empty((n_blocks, n_column), dtype='d')
for i in range(size_raw):
    if rank_raw == i:
        local_mult = blockMatrix@Omega1_local  # dim(n/p x l)

C_local = np.empty((n_blocks, n_column), dtype='d')
comm_raw.Reduce(local_mult, C_local, op=MPI.SUM, root=0)
C = np.empty((n, n_column), dtype='d')
comm_col.Gather(C_local, C, root=0)

root_to_consider = 0
if rank_raw == 0:
    root_to_consider = int(rank/int(sqrt(size)))

root_to_consider = comm_raw.bcast(root_to_consider, root=0)

Omega1_local = comm_raw.bcast(Omega1_local, root=root_to_consider)
local_mult2 = np.empty((n_column, n_column), dtype='d')
if rank_raw == 0:
    local_mult2 = np.transpose(Omega1_local)@C_local

B = np.empty((n_column, n_column), dtype='d')
comm_col.Reduce(local_mult2, B, op=MPI.SUM, root=0)

if rank == 0:
    print("Parallel C: ", C)

    C_ser = A @ Omega1
    print("Serial C: ", C_ser)

    print("Parallel B: ", B)

    B_ser = np.transpose(Omega1)@A@Omega1
    print("Serial B: ", B_ser)
