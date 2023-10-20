from mpi4py import MPI
import numpy as np

# Function to perform matrix-vector multiplication
def matrix_vector_multiplication(matrix, vector):
    result = matrix@vector
    return result

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

wt = MPI.Wtime() # We are going to time this

# Define the matrix and vector
cols = 4
rows = 8
num_rows_block = int(rows/size)
num_cols_block = int(cols/size)

matrix = None
matrix_to_send = None
vector = None
# Try changing dtype below to see what happens!
global_result = np.empty((rows, 1), dtype = 'int')

if rank == 0:
  matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20],
                   [21, 22, 23, 24],
                   [25, 26, 27, 28],
                   [29, 30, 31, 32]])
  # There are several ways of sending the matrix in the
  # order you want. This is very important since Python
  # is row major!
  arrs = np.split(matrix, size, axis=1)
  raveled = [np.ravel(arr) for arr in arrs]
  matrix_to_send = np.concatenate(raveled)
  vector = np.array([7, 8, 9, 10])

# Define the buffer where we are going to receive the block of the matrix
submatrix = np.empty((num_cols_block, rows), dtype = 'int')
local_vector = np.empty((num_cols_block, 1), dtype = 'int')
# Scatterv: Scatter Vector, scatter data from one process to all other
# processes in a group providing different amount of data and displacements
# at the sending side
comm.Scatterv(matrix_to_send, submatrix, root=0)
comm.Scatterv(vector, local_vector, root=0)
submatrix = np.transpose(submatrix)
print("rank: ", rank, "submatrix: ", submatrix, " local vector: ", local_vector)

# Compute local multiplication
local_result = matrix_vector_multiplication(submatrix, local_vector)

# Gather results on the root process
# Gatherv: Gather Vector, gather data to one process from all
# other processes in a group providing different amount of
# data and displacements at the receiving sides
comm.Reduce(local_result, global_result, op=MPI.SUM, root = 0)

# Print the result on the root process
if rank == 0:
    wt = MPI.Wtime() - wt
    print("Matrix:")
    print(matrix_to_send)
    print("Vector:")
    print(vector)
    print("Result:")
    print(global_result)
    print("Result with numpy: ")
    print(matrix@vector)
    print("Time taken to compute matrix vector multiplication: ")
    print( wt )
