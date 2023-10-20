from mpi4py import MPI
import numpy as np

# Testing what comm.Split() does

# Initialize MPI (world)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the matrix
n_blocks = 2
n = size
npr = 2

matrix = None
matrix_transpose = None

if rank%npr == 0:
  matrix = np.arange(1, n*n + 1, 1, dtype=int)
  matrix = np.reshape(matrix, (n,n))
  arrs = np.split(matrix, n, axis=1)
  raveled = [np.ravel(arr) for arr in arrs]
  matrix_transpose = np.concatenate(raveled)
  print(matrix)


comm_cols = comm.Split(color = rank/npr, key = rank%npr)
comm_rows = comm.Split(color = rank%npr, key = rank/npr)

# Get ranks of subcommunicator
rank_cols = comm_cols.Get_rank()
rank_rows = comm_rows.Get_rank()

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
    

print("Rank in original communicator: ", rank,
      " rank in splitrows: ", rank_rows, " rank in splitcols: ", rank_cols,
      "submatrix: ", submatrix,
      "block matrix: ", blockMatrix, "\n\n")



