from mpi4py import MPI
import numpy as np
from numpy.linalg import norm

# CSG (first attempt, just calculate Q)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

m = 3*size
n = 2*size
local_size = int(m/size) # Dividing by rows

# Define 
W = None
Q = None
QT = None
P = None
if rank == 0:
    W = np.arange(1, m*n + 1, 1, dtype = 'd')
    W = np.reshape(W, (m, n))
    W = W + np.eye(m, n) # Make this full rank
    Q = np.zeros((m,n), dtype = 'd')
    QT = np.zeros((n,m), dtype = 'd')
    P = np.eye( m, m, dtype = 'd') # first projector is just I

# In here: we first build Q and then we build R
# Decide what needs to be scattered/broadcast
W_local = 
q_local = 
QT_local = 
P_local = 
W_local = 
comm.Scatterv(P, P_local, root = 0)

# For the first column
q_local = P_local@W_local[:, 0]
# Normalize, put this column in Q (and row in QT)

# Start interating in the columns

for k in range(1, n):
    # We've already built column 0 so we move to column 1
    # First: we must build the projector P, using SUMMA
    localMult = # What needs to go here so that we can do a reduce?
    comm.Reduce(localMult, P, op = MPI.SUM, root = 0) # Projector
    comm.Scatterv(P, P_local, root = 0) # scatter rows of projector
    q_local = P_local@W_local[:, k] # project the k-th column of W
    # Normalize, put this column in Q (and row in QT)
    # Update the part of Q and QT that is in each processor
    comm.Scatterv(QT, QT_local, root = 0)

    
# Print in rank = 0
if( rank == 0):
    print("Q: \n", Q)

    

