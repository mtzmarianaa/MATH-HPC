from mpi4py import MPI
import numpy as np

# 2D distribution for matrix-vector multiplication

# Initialize MPI (world)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the matrix
n_blocks = 4
n = size*2
npr = 2
matrix = None
matrix_transpose = None
x = None
y = None
sol = None

if rank%npr == 0:
  matrix = np.arange(1, n*n + 1, 1, dtype=int)
  matrix = np.reshape(matrix, (n,n))
  arrs = np.split(matrix, n, axis=1)
  raveled = [np.ravel(arr) for arr in arrs]
  matrix_transpose = np.concatenate(raveled)
  x = np.arange(1, n+1, 1, dtype = int)
  x = np.reshape(x, (n,1) )
  sol = np.empty((n,1), dtype = int)

comm_cols = comm.Split(color = rank/npr, key = rank%npr)
comm_rows = comm.Split(color = rank%npr, key = rank/npr)

# Get ranks of subcommunicator
rank_cols = comm_cols.Get_rank()
rank_rows = comm_rows.Get_rank()

### DISTRIBUTE THE MATRIX
# Select columns
submatrix = np.empty((n_blocks, n), dtype = 'int')

# Then we scatter the columns and put them in the right order
receiveMat = np.empty((n_blocks*n), dtype = 'int')
comm_cols.Scatterv(matrix_transpose, receiveMat, root = 0)
subArrs = np.split(receiveMat, n_blocks)
raveled = [np.ravel(arr, order='F') for arr in subArrs]
submatrix = np.ravel(raveled, order = 'F')

# Then we scatter the rows
blockMatrix = np.empty((n_blocks, n_blocks), dtype = 'int')
comm_rows.Scatterv(submatrix, blockMatrix, root = 0)

### DISTRIBUTE X USING COLUMNS
x_block = np.empty((n_blocks, 1), dtype = 'int')
comm_cols.Scatterv(x, x_block, root = 0)

# Multiply in place each block matrix with each x_block
local_mult = blockMatrix@x_block

# Now sum those local multiplications along rows
rowmult = np.empty((n_blocks, 1), dtype = 'int')
comm_cols.Reduce(local_mult, rowmult, op = MPI.SUM, root = 0)

# # Now we gather all of this on the root process of the original comm
if rank_cols == 0:
    comm_rows.Gather(rowmult, sol, root = 0)

# Print in the root process
if(rank == 0):
    print("Solution with MPI: ", np.transpose(sol),
          "Solution with Python: ", np.transpose(matrix@x))

