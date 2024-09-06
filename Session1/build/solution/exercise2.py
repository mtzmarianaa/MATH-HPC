# Exercise II: Function in Python
import numpy as np
from time import perf_counter as tic

x = np.array( [1e-10, 1e-12, 1e-14, 1e-16] )

# Function 1: direct

def f_direct(x):
    '''
    Implementation of the following function:
        ( sqrt(1 + x) - 1 )/x
    Parameters
    ----------
    x : fl, np.array
        Number or np array where to evaluate f(x)

    Raises
    ------
    ZeroDivisionError
        If x=0
    '''
    if np.any(x == 0):
        raise ZeroDivisionError("Can't divide by zero!")
    return (np.sqrt(1 + x) - 1)/x

def f_opt1(x):
    '''
    Implementation of the following function:
        ( sqrt(1 + x) - 1 )/x
    written as
        1/(sqrt(1 + x) + 1)
    Parameters
    ----------
    x : fl, np.array
        Number or np array where to evaluate f(x)
    '''
    return 1/(np.sqrt(1 + x) + 1)

def f_opt2(x):
    '''
    Implementation of the following function:
        ( sqrt(1 + x) - 1 )/x
    written as its O(x^4) approximation
       1/2 - x/8 + x^2/16 - 5*x^3/128
    Parameters
    ----------
    x : fl, np.array
        Number or np array where to evaluate f(x)
    '''
    return 0.5 - x/8 + x**2/16 - 5*x**3/128

print("\n\nEvaluating the functions using a for loop \n")
f_eval_for_0 = np.empty_like(x)
f_eval_for_1 = np.empty_like(x)
f_eval_for_2 = np.empty_like(x)
t0 = tic()
for i in range(len(x)):
    f_eval_for_0[i] = f_direct(x[i])
    f_eval_for_1[i] = f_opt1(x[i])
    f_eval_for_2[i] = f_opt2(x[i])
t1 = tic() - t0
print("Time taken for loop: " + f"{t1:.2e}" + " s.")

print("\n\nEvaluating the functions using numpy vector algebra \n")
t0 = tic()
f_eval_np_0 = f_direct(x)
f_eval_np_1 = f_opt1(x)
f_eval_np_2 = f_opt2(x)
t1 = tic() - t0
print("Time taken numpy vector algebra: " + f"{t1:.2e}" + " s.")

print("\n\nSolutions given by different implementations:")
print("Direct:   " + str(f_eval_np_0))
print("Method 1: " + str(f_eval_np_1))
print("Method 2: " + str(f_eval_np_2))


