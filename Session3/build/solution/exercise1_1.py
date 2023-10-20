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

matrix = None
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
  vector = np.array([7, 8, 9, 10])

# Define the buffer where we are going to receive the block of the matrix
submatrix = np.empty((num_rows_block, cols), dtype='int')
# Scatterv: Scatter Vector, scatter data from one process to all other
# processes in a group providing different amount of data and displacements
# at the sending side
comm.Scatterv(matrix, submatrix, root=0)
vector = comm.bcast(vector, root = 0)

# Compute local multiplication
local_result = matrix_vector_multiplication(submatrix, vector)

# Gather results on the root process
# Gatherv: Gather Vector, gather data to one process from all
# other processes in a group providing different amount of
# data and displacements at the receiving sides
comm.Gatherv(local_result, global_result, root = 0)

# Print the result on the root process
if rank == 0:
    wt = MPI.Wtime() - wt
    print("Matrix:")
    print(matrix)
    print("Vector:")
    print(vector)
    print("Result:")
    print(global_result)
    print("Time taken: ")
    print( wt )

