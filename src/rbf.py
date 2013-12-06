import numpy as np
from random import random
from utils import Point, null, column
DEBUG = True
def affpoints(xk, D, theta, delta):
    m = len(D)
    n = len(D[0].x)
    Y = [Point(n)]
    Y2 = [xk]
    index = [-1]
    Z = np.matrix(np.identity(n))
    p = int(round(m*random()+0.5))
    if p > 1.0:
        ## indexes check properly
        vec = range(p, m)
        vec.extend(range(0, p-1))
    else:
        vec = range(m)
    if DEBUG:
        print vec
    
    for i in vec:
        if np.linalg.norm(D[i].x - xk.x) <= delta:
            Z = np.matrix(Z)
            proj_z = Z*np.linalg.inv(Z.transpose()*Z)*Z.transpose()*np.matrix(D[i].x-xk.x).transpose()
            if np.linalg.norm(proj_z) >= delta*theta:
                point = Point(n)
                point.x = D[i].x - xk.x
                Y.append(point) 
                Y_matrix = array([])
                for j in range(len(Y)):
                    Y_matrix = np.concatenate((Y_matrix, Y[j].x), 1)
                Z = null(Y_matrix)
                Y2.append(D[i])
                index.append(i)
                print "here"
    if len(Y) == n + 1:
        linear = True
    else:
        linear = False
    return Y2, linear, Z, index

def rbfmodel(objective, fid, xk, fxk, krig, theta_1, theta_2, theta_3, theta_4, \
                 delta, deltamax, pmax, optTheta):
    deltamax = delta
    D = krig.points
    Y2, linear, Z, index = affpoints(xk, D, theta_1, theta_3*delta)
    if not linear:
        Z = Z*delta
        for i in range(len(Z[0,:])):
            # Hard code alert
            point = Point(2)
            point.x = xk.x + Z.view(np.ndarray)[:,i]
            Y2.append(point)
            D.append(point)
            f_high = objective(point, fid[1])
            point.fval = f_high - objective(point, fid[0])
            krig.points.append(point)
            index.append(len(krig.points)-1)
        linear = True

    krig = addpoints(xk, fxk[1]-fxk[0], D, Y2, krig, index, theta_2, theta_4*deltamax, pmax, optTheta)

    return krig, linear

def addpoints(xk, fxk, D, Y2, krig, index, theta, delta, pmax, opt_theta):
    points = krig.points
    # index(find(index=-1,1)) = length(fvals) + 1
    # hard code WARNING
    point = Point(2)
    point.x = xk.x
    point.fval = fxk
    points.append(point)
    Y2_0 = Y2
    index_0 = index
    
    if opt_theta:
        raise AssertionError("Optimization of hyperparameters not implemented.")

    else:
        ntest = 0
        psi = 1.0
        thetagauss = krig.params[0]


    L = []
    Y2 = Y2_0
    index = index_0

    return krig


    
