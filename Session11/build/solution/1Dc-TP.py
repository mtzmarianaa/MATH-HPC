# Implementation of deterministic column selection: tournament pivoting


from mpi4py import MPI
import numpy as np
from numpy.linalg import norm, svd
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.linalg import hadamard
from math import sqrt, exp, ceil, log
import pandas as pd

# Custom library
from sRRQR import sRRQR_rank

plt.ion()

# Functions to build the matrix A (see last week's exercises)
def readData(filename, size = 784, save = False):
    '''
    Read MNIST sparse data from filename
    and transforms this into a dense
    matrix, each line representing an entry
    of the database (i.e. a "flattened" image)
    '''
    dataR = pd.read_csv(filename, sep=',', header = None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))
    # Format accordingly
    for i in range(n):
        l = dataR.iloc[i, 0]
        labels[i] = int(l[0]) # We know that the first digit is the label
        l = l[2:]
        indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()]
        # Separate indices and values
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        # Fill in the values at the specified indices
        data[i, indices] = values
    if save:
        data.tofile('./denseData.csv', sep = ',',format='%10.f')
        labels.tofile('./labels.csv', sep = ',',format='%10.f')
    return data, labels

def buildA_sequential(data, c = 1e3, save = False):
    '''
    Function to build A out of a data base
    using the RBF exp( -||x_i - x_j||/c)
    Notice that we only need to fill in the
    upper triangle part of A since it's symmetric
    and its diagonal elements are all 1.
    '''
    n = data.shape[0]
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(j):
            A[i,j] = exp( -norm( data[i, :] - data[j, :])**2/c)
    A = A + np.transpose(A)
    np.fill_diagonal(A, 1.0)
    if save:
        A.tofile('./A.csv',sep=',',format='%10.f')
    return A

def matTranspose(A, n):
    '''
    Since we want to distribute A's columns
    we need to manipulate our data into the correct
    order before sending it.
    '''
    arrs = np.split(A, n, axis = 1)
    raveled = [np.ravel(arr) for arr in arrs]
    A_transpose = np.concatenate(raveled)
    return A_transpose

# Initialize MPI (world)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = None
m = None
n = None
local_sizeCols = None
A_T = None
f = 1.05
k = 4

if rank == 0:
    # Read the files and build the matrix A here (see last week's exercises)
    filename = "mnist_780"
    data, labels = readData(filename)
    A = buildA_sequential(data)
    #A = A[0:16, 0:16]
    # D = np.diag([1, 0.9, 0.8, 0.7, 0.65, 0.4, 0.1, 0.001])
    # H = hadamard(8)
    # H = 1/norm(H[:, 0])*H
    # A = H@D@np.transpose(H)
    # A = np.arange(0, 108)
    # A = np.reshape(A, (9, 12)) + 0.0
    m = A.shape[0]
    n = A.shape[1]
    print("m: ", m, "n: ", n)
    local_sizeCols = n//size # number of columns
    A_T = matTranspose(A, n)

# Step one: partition A into 4 column blocks
local_sizeCols = comm.bcast(local_sizeCols, root=0)
m = comm.bcast(m, root=0)
A_local = np.empty((local_sizeCols, m))
comm.Scatter(A_T, A_local, root = 0)
# Step 0
# From each column block A1i, i = 1, . . . , 4, k columns are selected by using
# strong RRQR, and their indices are given in Ii0.
Q, R, P = sRRQR_rank(np.transpose(A_local), f, k)
P = P[0:k]
Pall = deepcopy(P) + rank*local_sizeCols
#Start iterating
for i in range(1, ceil(log(size)) + 1):
    #print("i: ", i)
    if rank%(2**i) == 0 and rank + 2**(i-1)<size:
        j = rank + 2**(i-1)
        Phere = np.copy(P)
        Psave = deepcopy(Pall)
        Ahere = np.copy(A_local)
        Ahere = Ahere[Phere, :]
        # We receive the first k indices from Pj from processor j
        #print("Receiving and factorizing at: ", rank)
        comm.Recv(P, source = j, tag = 77)
        comm.Recv(A_local, source = j, tag = 88)
        comm.Recv(Pall, source = j, tag = 99)
        A_local = A_local[P, :]
        A_local = np.concatenate((Ahere, A_local))
        Pall = np.concatenate((Psave, Pall))
        # strong RRQR
        Q, R, P = sRRQR_rank(np.transpose(A_local), f, k)
        P = P[0:k]
        Pall = Pall[P]
    elif rank%(2**i) == 2**(i-1):
        # Send the local P
        #print("Sending from: ", rank)
        comm.Send(P, dest = rank - 2**(i-1), tag = 77)
        comm.Send(A_local, dest = rank - 2**(i-1), tag = 88)
        comm.Send(Pall, dest = rank - 2**(i-1), tag = 99)


if rank == 0:
    # Print the selected columns I02
    # Make the low rank approximation
    Pall = Pall.flatten()
    print("Selected columns: ", Pall)
    Atilde = A[:, Pall]
    U, Sigma, V = svd(Atilde)
    U = U[:, 0:k]
    Sigma = Sigma[0:k]
    V = V[0:k, 0:k]
    appsRRQR = U@np.diag(Sigma)@V
    # Singular values of full A
    Uf, Sigmaf, Vf = svd(A)
    Uf = Uf[:, 0:k]
    Sigmaf = Sigma[0:k]
    Vf = Vf[0:k, 0:k]
    appfull = Uf@np.diag(Sigmaf)@Vf
    # Compute the theoretical bound
    gamma = sqrt( 1 + f**2*k*(n-k) )
    print("L2 error with truncated SVD: ",
          norm( appsRRQR - appfull)  )
    print(r'$\gamma (n, k) = $')
    print(gamma)
    print(r'$\sigma_i(A)/\sigma_i(R_{11}) = $')
    print(np.divide(Sigma,Sigmaf))

