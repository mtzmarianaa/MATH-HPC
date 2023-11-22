import numpy as np
from numpy.linalg import svd, qr, norm
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from math import log, sqrt, floor
import torch
from hadamard_transform import hadamard_transform

# So that the plots are "interactive" when we run this script
plt.ion()

def SVD_Blanczos(A, l, i, Sigma_full = False):
    '''
    From Rokhlin, Szlam, Tygert paper A Randomized Algorithm For Principal Component Analysis, algorithm 4.4 called Blanczos
    IN :   
           A          : mxn matrix to be factorized
           l          : paramter for our approximated matrix C of size mxc
           i          : order of approximation wanted
           Sigma_full : transition probabilities
    OUT :
           U          : approximated left singular vectors
           Sigma      : approximated singular values
           V          : approximated right singular vectors
    ''' 
    m = A.shape[0]
    n = A.shape[1]
    # STEP 1
    # Using a random number generator form a real lxm matrix G whose entries are iid Gaussian and compute the lxn matrices
    G = np.random.normal(loc= 0.0, scale = 1.0, size = [l, m])
    R = np.zeros(( (i+1)*l, n ))
    R_temp = G@A
    R[0:(l), :] = R_temp
    for j in range(i):
        R_temp = R_temp@np.transpose(A)@A
        R[ (j+1)*l:(j+2)*(l), : ] = R_temp
    # STEP 2
    # Using QR decomposition form a real n x ((i+1)l) matrix Q whose columns are orthonormal
    Q, S = qr(np.transpose(R))
    # STEP 3
    # Compute the m x ( (i+1)l ) product matrix
    T = A@Q
    # STEP 4
    # Form an SVD of T
    U, Sigma, Wt = svd(T)
    # STEP 5
    # Compute the n x( (i+1)l ) product matrix
    V = Q@np.transpose(Wt)
    if Sigma_full:
        S_t = Sigma
        Sigma = np.zeros((m,n))
        np.fill_diagonal(Sigma, S_t)
    return U, Sigma, V


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
l = k + 12
i = 1
sigma_k1S = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
errorApprox = np.empty(6)
errorApproxRel = np.empty(6)

for s in range(6):
    sigma = sigma_k1S[s]
    # Build A
    A, sigmas = buildA(m, sigma, k)
    # Blanczos
    U, S, V = SVD_Blanczos(A, l, i)
    Sigma = np.zeros((U.shape[1], S.shape[0]))
    np.fill_diagonal(Sigma, S)
    # Plot the decay of the singular values
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(np.arange(m), sigmas, marker = 'o', c = "#0800ff")
    plt.title("Decay on singular values for " + r"$\sigma_{k+1} = $" + str(sigma))
    plt.xlabel("k")
    plt.ylabel(r"$\sigma_{k}$")
    # Save the error of the approximation || A - U \Sigma V^\top ||
    errorApprox[s] = norm( A - U@Sigma@np.transpose(V), 'fro')
    errorApproxRel[s] = errorApprox[s]/norm(A, 'fro')

# Plot the errors of the approximation
plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(sigma_k1S, errorApprox, marker = 'o', c = "#8700ff")
plt.title("Errors in approximation, different  " + r"$\sigma_{k+1}$")
plt.xlabel(r"$\sigma_{k+1}$")
plt.ylabel(r"$\| A - U \Sigma V^\top\|$")

# Plot relative errors of the approximation
plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(sigma_k1S, errorApproxRel, marker = 'o', c = "#ff8f00")
plt.title("Relative errors in approximation, different  " + r"$\sigma_{k+1}$")
plt.xlabel(r"$\sigma_{k+1}$")
plt.ylabel(r"$\| A - U \Sigma V^\top\|$")

