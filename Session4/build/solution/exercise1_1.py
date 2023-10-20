from mpi4py import MPI
import numpy as np
from numpy.linalg import norm

# CSG

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

wt = MPI.Wtime() # We are going to time this

m = 50*size
n = 10*size
local_size = int(m/size)

# Define 
W = None
Q = None
R = None
Qkreceived = None
QT = None
P = None
if rank == 0:
    W = np.arange(1, m*n + 1, 1, dtype = 'd')
    W = np.reshape(W, (m, n))
    W = W + np.eye(m, n) # Make this full rank
    Q = np.zeros((m,n), dtype = 'd')
    QT = np.zeros((n,m), dtype = 'd')
    Qkreceived = np.zeros((m, 1), dtype = 'd')
    R = np.zeros((n,n), dtype = 'd')
    P = np.eye( m, m, dtype = 'd')

# In here: we first build Q and then we build R
W_local = np.zeros((local_size, n), dtype = 'd')
q_local = np.zeros((local_size, 1), dtype = 'd')
QT_local = np.zeros((local_size, m), dtype = 'd')
P_local = np.zeros((local_size, m), dtype = 'd')
W_local = comm.bcast(W, root = 0)
comm.Scatterv(P, P_local, root = 0)

# For the first column
q_local = P_local@W_local[:, 0]

# Normalize
comm.Gather(q_local, Qkreceived, root = 0)
if( rank == 0):
    col = Qkreceived[:, 0]/norm(Qkreceived)
    Q[:, 0] = col
    QT[0, :] = col
comm.Barrier()
comm.Scatterv(QT, QT_local, root = 0) # We have COLUMNS of Q (or rows of Qt)
# Start interating in the columns

for k in range(1, n):
    # We've already built column 0 so we move to column 1
    # First: we must build the projector P, using SUMMA
    localMult = 1/size*np.eye(m,m) - np.transpose(QT_local)@QT_local
    comm.Reduce(localMult, P, op = MPI.SUM, root = 0) # Projector
    comm.Scatterv(P, P_local, root = 0) # scatter rows of projector
    q_local = P_local@W_local[:, k]
    # Normalize
    comm.Gather(q_local, Qkreceived, root = 0)
    if(rank == 0):
        col = Qkreceived[:, 0]/norm(Qkreceived)
        Q[:, k] = col
        QT[k, :] = col
    comm.Barrier()
    comm.Scatterv(QT, QT_local, root = 0)

# Compute R as R = QtA
W_rows = np.zeros((local_size, n), dtype = 'd')
Q_local = np.zeros((local_size, n), dtype = 'd')
comm.Scatterv(W, W_rows, root = 0)
comm.Scatterv(Q, Q_local, root = 0)
localMult_R = np.transpose(Q_local)@W_rows
comm.Reduce(localMult_R, R, op = MPI.SUM, root = 0)

    
# Print in rank = 0
if( rank == 0):
    wt = MPI.Wtime() - wt
    print("Size of W: ", W.shape)
    print("Orthogonality of Q: ", norm(np.eye(n) - np.transpose(Q)@Q))
    print("Q: \n", Q.shape)
    print("R: \n", R.shape)
    print("Time taken: ", wt)

    

