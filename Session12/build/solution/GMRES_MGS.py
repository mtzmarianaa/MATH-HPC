# Algorithm 2 or GMRES with Modified Gram Shmidt

import numpy as np
from numpy.linalg import lstsq, norm


def arnoldi_cgs(A, b, x0, m, r0 = None, beta = None, q = None):
    '''
    Algorithm 1, Arnoldi
    '''
    if r0 is None and beta is None and q is None:
        r0 = b - A@x0
        beta = norm(r0)
        q = r0/beta
    # Define
    Hbar = np.zeros((m+1, m))
    Q = np.zeros((A.shape[1], m))
    Q[:, 0] = q
    for j in range(0, m):
        w = A@q
        hij = [np.dot(w, Q[:, i]) for i in range(j+1)]
        Hbar[0:(j+1), j] = hij
        for i in range(j+1):
            w = w - hij[i]*Q[:, i]
        Hbar[j+1, j] = norm(w)
        if Hbar[j+1, j]<1e-10:
            break
        q = w/Hbar[j+1, j]
        Q[:, j+1] = q
    H = Hbar[0:m, :]
    return H, Q

def mgs(W, Q = None):
    '''
    Modified Gram Schmidt. If Q is not none,
    we want to orthogonalize the columns of V with respect
    to the columns in Q. We assume Q is orthonormal
    '''
    if(len(W.shape) == 1):
        W = np.reshape(W, (W.shape[0], 1))
    n = W.shape[0]
    m = W.shape[1]
    if Q is None:
        start = 1
        Qm = np.empty_like(W)
        end = m
        Qm[:, 0] = W[:,0]/norm(W[:, 0])
    else:
        start = Q.shape[1]
        Qm = np.empty( (n, m+start) )
        Qm[:, 0:start] = Q
        end = m+start
    # Start the iteration
    for i in range(start, end):
        Qm[:, i] = W[:, i-start]
        for j in range(0, i):
            Qm[:, i] = Qm[:, i] - np.dot( Qm[:, j], Qm[:, i] )*Qm[:, j]
        Qm[:, i] = Qm[:, i]/norm(Qm[:, i])
    return Qm

def gmres_mgs(A, b, x0, m, tol):
    '''
    GMRES algorithm with MGS, algorithm 2 from slides 
    '''
    x = x0
    r = b - A@x
    beta = norm(r)
    q = r/beta
    e1 = np.zeros((m+1, ))
    e1[0] = 1.0
    # Define
    Hbar = np.zeros((m+1, m))
    Q = np.zeros((A.shape[1], m+1))
    Q[:, 0] = q
    for j in range(m):
        print(j)
        w = A@q
        # Arnoldi
        hij = [np.dot(w, Q[:, i]) for i in range(j+1)]
        Hbar[0:(j+1), j] = hij
        for i in range(j+1):
            w = w - hij[i]*Q[:, i]
        Hbar[j+1, j] = norm(w)
        # Least squares
        ym = lstsq( Hbar[:, 0:(j+1)], beta*e1 )[0]
        x = x0 + Q[:, 0:(j+1)]@ym
        if norm( A@x - b  )<tol:
            print("Residual tolerance reached")
            break
        if Hbar[j+1, j]<1e-12:
            print("Hbar[j+1, j] too small")
            break
        q = w/Hbar[j+1, j]
        Q[:, j+1] = q
    return x, j, norm(A@x - b)
        
    
