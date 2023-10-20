# Implementation of parallel TSQR. We asume 4 processors

from mpi4py import MPI
import numpy as np
from numpy.linalg import norm, qr
from math import log, ceil
from scipy.sparse import block_diag

# Initialize MPI (world)
comm = MPI.COMM_WORLD
    

def my_tsqr(A, comm, root = 0):
    '''
    Naive implementation of parallel TSQR.
    We assume that A is mxn and m is divisible
    by the number of processors. In this implementation
    we are computing the reduced QR at each iteration.
    '''
    rank = comm.Get_rank()
    size = comm.Get_size()
    m = A.shape[0]
    n = A.shape[1]
    local_size = m//size # Number of 
    dtype = A.dtype # Get type of variables in A
    localQks = [] # Here we are going to save the local Qks
    comm.Barrier()
    # Step 0: compute the QR factorization of each block of rows
    wt = MPI.Wtime() # We are going to time this
    A_local = np.zeros((local_size, n), dtype = dtype)
    comm.Scatterv(A, A_local, root = root) # Get the block rows
    Qrk, Rk = qr(A_local) # Local QR at root
    localQks.append(Qrk)
    # Start iterating
    for k in range(1, ceil(log(size))+1):
        print("k:", k)
        if rank%(2**k) == root and rank + 2**(k-1)<size:
            j = rank + 2**(k-1)
            # We receive Rj from procesor j
            RkHere = np.copy(Rk)
            print("Receiving and factorizing at: ", rank)
            comm.Recv(Rk, source = j, tag = 77)
            RtoFactorize = np.concatenate((RkHere, Rk))
            Qrk, Rk = qr( RtoFactorize )
            localQks.append(Qrk)
            print("Size of RtoFactorize: ", RtoFactorize.shape)
            print("Size of Qrk: ", Qrk.shape, "\n")
        elif rank%(2**k) == 2**(k-1):
            # Send the local Rk
            print("Sending from: ", rank)
            comm.Send(Rk, dest = rank - 2**(k-1), tag = 77)
    comm.Barrier()
    print("Rank: ", rank, " len local Qks: ", localQks, '\n\n')
    comm.Barrier()
    # Get Q explicitly
    Q = None
    if rank == 0:
        Q = np.eye(n, n)
        Q = localQks[-1]@Q
        localQks.pop()
    # Start iterating through the tree backwards
    for k in range(ceil(log(size))-1, -1, -1):
        print("k: ", k)
        color = rank%(2**k)
        key = rank//(2**k)
        comm_branch = comm.Split(color = color, key = key)
        rank_branch = comm_branch.Get_rank()
        print("Rank: ", rank, " color: ", color, " new rank: ", rank_branch)
        if( color == 0):
            # We scatter the columns of the Q we have
            Qrows = np.empty((n,n), dtype = 'd')
            comm_branch.Scatterv(Q, Qrows, root = 0)
            # Local multiplication
            print("size of Qrows: ", Qrows.shape)
            Qlocal = localQks[-1]@Qrows
            print("size of Qlocal: ", Qlocal.shape)
            localQks.pop()
            # Gather
            Q = comm_branch.gather(Qlocal, root = 0)
            if rank == 0:
                Q = np.concatenate(Q, axis = 0)
                print(Q.shape)
        comm_branch.Free()
    # Then we save
    if rank == 0:
        R = np.copy(Rk)
        wt = MPI.Wtime() - wt
        I = np.eye(m)
        Qtrue, Rtrue = qr(A)
        print("Time taken: ", wt)
        #print("Measure of orthogonality: ", norm( np.eye(m, m) - Q@np.transpose(Q)))
        print("Comparison with numpy: \n for Q: ", "\n for R: ",
              R - Rtrue)
        print("Test of orthogonality: ", norm(I - Q@np.transpose(Q)))
        #return R, wt

rank = comm.Get_rank()
size = comm.Get_size()
m = 8*size
n = size
A = np.arange(1, m*n + 1, 1, dtype = 'd')
A = np.reshape(A, (m, n))
A = A + np.eye(m,n)
my_tsqr(A, comm)

