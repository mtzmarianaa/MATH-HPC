# Randomized GS from
# https://arxiv.org/pdf/2011.05090.pdf

import numpy as np
from numpy.linalg import lstsq, norm
from numpy.random import normal
from math import ceil

def rand_GS(W, k = None, Q = None, Omega = None):
    '''
    Randomized Gram Schmidt. Q is a matrix
    with orthonormal columns, we want to orthogonalize
    the columns of W against the columns of Q.
    Omega is the sketching matrix, if not given
    assumed to be normal
    '''
    n = W.shape[0]
    m = W.shape[1]
    if k is None:
        k = ceil(0.3*min(m,n))
    if Omega is None:
        Omega = normal(loc = 0.0, scale = 1.0, size = (k, n))
    if Q is None:
        start = 1
        Qm = np.empty_like(W)
        end = m
        Sm = np.empty( (k,m) )
        Qm[:, 0] = W[:,0]/norm(W[:,0])
        Sm[:, 0] = Omega@W[:,0]
        Sm[:, 0] = Sm[:,0]/norm(Sm[:,0])
    else:
        start = Q.shape[1]
        Qm = np.empty( (n, m+start) )
        Qm[:, 0:start] = Q # Since these columns are orthonormal already
        end = m+start
        Sm = np.empty( (k, m+start) )
        Sm[:, 0:start] = Omega@Q
    # Start the iteration
    for i in range(start, end):
        pi = Omega@W[:, i-start]
        # Solve the least squares problem
        ri = lstsq(Sm[:, 0:i], pi)[0]
        qiPrime = W[:, i-start] - Qm[:, 0:i]@ri
        siPrime = Omega@qiPrime
        rii = norm(siPrime)
        Sm[:, i] = siPrime/rii
        Qm[:, i] = qiPrime/rii
    return Qm
    
        




