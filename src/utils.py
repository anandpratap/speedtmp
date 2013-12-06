import numpy as np

def column(matrix, i):
    print "Column"
    c = []
    for row in matrix:
        print row
    return [row[i] for row in matrix]
def null(A, eps=1e-15):
    """   
    http://mail.scipy.org/pipermail/scipy-user/2005-June/004650.html
    """
    u, s, vh = np.linalg.svd(A)
    n = A.shape[1]   # the number of columns of A
    if len(s)<n:
        expanded_s = np.zeros(n, dtype = s.dtype)
        expanded_s[:len(s)] = s
        s = expanded_s
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)

def rosenbrock(point, fid):
    """
    Objective function to be minimized

    Inputs:
       x   <type:array>  array of design variables
       fid <type:int>    fidelity level higher is better

    Output:
       <type:float> function value
    """
    x = point.x
    if fid == 1:
        return 3 + x[0]*x[0] + x[1]*x[1]
    elif fid == 2:
        return (x[1]-x[0])**2 + (1-x[0])**2

class Point(object):
    def __init__(self, n):
        self.x = np.zeros(n)
        self.fval = None
