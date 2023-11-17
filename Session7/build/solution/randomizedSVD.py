import numpy as np
from numpy.linalg import svd, qr, norm
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from math import log, sqrt, floor
import torch
from hadamard_transform import hadamard_transform

# So that the plots are "interactive" when we run this script
plt.ion()

def SVD_rand(A, k, p):
    '''
    Randomized SVD with q = 1
    IN :   
           A          : mxn matrix to be factorized
           k          : order of approximation
           p          : such that l = p + k
    OUT :
           U          : approximated left singular vectors
           Sigma      : approximated singular values
           V          : approximated right singular vectors
    ''' 
    m = A.shape[0]
    n = A.shape[1]
    l = p+k
    # STEP 1
    # Using a random number generator form a i.i.d. Gaussian matrix
    Omega1 = np.random.normal(loc= 0.0, scale = 1.0, size = [n, l])
    Y = (A@np.transpose(A))@A@Omega1
    # Construct Q1
    Q1, R = qr(Y)
    # Compute B
    B = np.transpose(Q1)@A
    # Compute th rank-k truncated SVD of B
    U, Sigma, V = svd(B)
    U = U[:, 0:k]
    Sigma = Sigma[0:k]
    V = V[:, 0:k]
    U = Q1@U
    return Q1, U, Sigma, V


def buildA(m, sigma_k1, k = 10):
    '''
    From Rokhlin, Szlam, Tygert paper A Randomized Algorithm For Principal Component
    Analysis, build test matrix A of size mx(2m). We use the fast Hadamard transform
    IN:  m        : number of desired rows in matrix A
         sigma_k1 : (k+1)th biggest singular value of A
         k        : where we are going to truncate the approximation of A
    OUT: A        : matrix with desired structure

    QUESTION: Can we build A faster? Notice that Sigma is just a diagonal matrix.
    Also notice that we can use the fast Hadamard transform to build A.
    If you can, change this function so that it builds A faster!
    '''
    U = (1/sqrt(m))*hadamard(m)
    V = (1/sqrt(2*m))*hadamard(2*m)
    firstSig = [sigma_k1**(floor(j/2)/5) for j in range(1, k+1)]
    sigmas = firstSig + [sigma_k1*(m - j)/(m - 11) for j in range(k+1, m+1)]
    Sigma = np.zeros((m, 2*m))
    np.fill_diagonal(Sigma, sigmas)
    return U@Sigma@np.transpose(V), sigmas

# Test
m = 2**11
k = 10
p = 6
sigma_k1S = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
errorApprox = np.empty(6)
errorApproxRel = np.empty(6)
errTh = np.empty(6)

for s in range(6):
    sigma = sigma_k1S[s]
    # Build A
    A, sigmas = buildA(m, sigma, k)
    # Randomized SVD
    Q1, U, S, V = SVD_rand(A, k, p)
    Sigma = np.zeros((U.shape[1], S.shape[0]))
    np.fill_diagonal(Sigma, S)
    # Plot the decay of the singular values
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(np.arange(m), sigmas, marker = 'o', c = "#0800ff")
    plt.title("Decay on singular values for " + r"$\sigma_{k+1} = $" + str(sigma))
    plt.xlabel("k")
    plt.ylabel(r"$\sigma_{k}$")
    # Save the error of the approximation
    errTh[s] = norm( A - Q1@np.transpose(Q1)@A )


# Plot error from theorem
plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(sigma_k1S, errTh, marker = 'o', c = "#ff8f00")
plt.title(r"$| \| A - Q_1Q_1^{\top}A\|  $" + " and decay on singular values")
plt.xlabel(r"$\sigma_{k+1}$")
plt.ylabel(r"$| \| A - Q_1Q_1^{\top}A\|  $")


