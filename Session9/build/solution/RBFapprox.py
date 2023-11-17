import numpy as np
import matplotlib.pyplot as plt
from math import exp, ceil, log
import pandas as pd
from numpy.linalg import norm, qr, cholesky, inv, svd, matrix_rank, lstsq, cond

plt.ion()

def readData(filename, size = 784, save = True):
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
        

# Define function to build A
def buildA_sequential(data, c = 1e-4, save = True):
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

# We are going to use the previously build
# function randNystrom
def randNystrom(A, Omega, returnExtra = True):
    '''
    Randomized Nystrom
    Option to return the singular values of B and rank of A
    '''
    m = A.shape[0]
    n = A.shape[1]
    l = Omega.shape[1]
    C = A@Omega
    B = np.transpose(Omega)@C
    try:
        # Try Cholesky
        L = cholesky(B)
        Z = lstsq(L, np.transpose(C))[0]
        Z = np.transpose(Z)
    except np.linalg.LinAlgError as err:
        # Do LDL Factorization
        lu, d, perm = ldl(B)
        # Question for you: why is the following line not 100% correct? 
        lu = lu@np.sqrt(np.abs(d))
        # Does this factorization actually work?
        L = lu[perm, :]
        Cperm = C[:, perm]
        Z = lstsq(L, np.transpose(Cperm))[0]
        Z = np.transpose(Z)
    Q, R = qr(Z)
    U_t, Sigma_t, V_t = svd(R)
    Sigma_t = np.diag(Sigma_t)
    U = Q@U_t
    if returnExtra:
        S_B = cond(B)
        rank_A = matrix_rank(A)
        return U, Sigma_t@Sigma_t, np.transpose(U), S_B, rank_A
    else:
        return U, Sigma_t@Sigma_t, np.transpose(U)

# Try solving the least squares problem with randomized Nystrom
filename = "mnist_780"
n_omega = 2048
l = 50
cs = [1e1, 1e2, 1e3, 1e4, 1e5]
Omega = np.random.normal(loc= 0.0, scale = 1.0, size = [n_omega, l])
data, labels = readData(filename, save = False)

err_cN = np.zeros((5, 1))
err_cE = np.zeros((5, 1))

for i in range(len(cs)):
    c = cs[i]
    A = buildA_sequential(data, c = c, save = False)
    U, Sigma, V_t = randNystrom(A, Omega, returnExtra = False)
    # Solve the least squares problem
    S_rec = np.where(Sigma>1e-10, 1/Sigma, 0)
    lam = np.transpose(V_t)@S_rec@np.transpose(U)@labels
    err_cN[i] = norm( A@lam - labels, 'nuc')/norm(A, 'nuc')

# Plot

plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(cs, err_cN, c = "#003aff", marker = 'o', label = 'Nystrom approx')
plt.legend()
plt.title("RBF approximation " + r'$\phi\left( \|x_i - x_j\| \right) = e^{- \|x_i - x_j\| / c}$')
plt.xlabel("c")
plt.ylabel("Relative error, nuclear norm")



# Now solve the problem with different values of l
ls = [10, 25, 50, 75, 100]
c = 100
data, labels = readData(filename, save = False)
A = buildA_sequential(data, c = c, save = False)

err_cN2 = np.zeros((5, 1))

for i in range(len(ls)):
    l = ls[i]
    Omega = np.random.normal(loc= 0.0, scale = 1.0, size = [n_omega, l])
    U, Sigma, V_t = randNystrom(A, Omega, returnExtra = False)
    # Solve the least squares problem
    S_rec = np.where(Sigma>1e-10, 1/Sigma, 0)
    lam = np.transpose(V_t)@S_rec@np.transpose(U)@labels
    err_cN2[i] = norm( A@lam - labels, 'nuc')/norm(A, 'nuc')

# Plot

plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(ls, err_cN2, c = "#003aff", marker = 'o', label = 'Nystrom approx')
plt.legend()
plt.title("RBF approximation " + r'$\phi\left( \|x_i - x_j\| \right) = e^{- \|x_i - x_j\| / c}$')
plt.xlabel("l")
plt.ylabel("Relative error, nuclear norm")



