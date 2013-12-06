import numpy as np
from rbf import rbfmodel
from utils import rosenbrock, Point


class Kriging(object):
    def __init__(self, corr, params, points):
        self.corr = corr
        self.params = params
        self.points = points

        self.Y2 = np.array([])
        self.Coeffs = np.array([])
        self.L = np.array([])
        self.sigma_2 = 0.0


class Speed(object):
    """
    Base class for optimizer, contains all the parameters and routines 
    
    Parameters:
       objective   <type:object>   Handle to the objective function
       n           <type:int>      Number of design variables
       LB          <type:array(n)> Lower bounds on design variables
       UB          <type:array(n)> Upper bounds on design variables
       
       fid         <type:array(nfid)> array of fidelity levels
       pmax        <type:int> maximum number of calibration points
       
       eta_0       <type:float> trust region movement criteria
       eta_1       <type:float> trust region expansion criteria
       gamma_0     <type:float> trust region contraction ratio
       gamma_1     <type:float> trust region expansion ratio
       delta_0     <type:float> initial trust region size
       delta_max   <type:float> maximum trust region size
       epsilon     <type:float> convergence tolerance
       kd          <type:float> fraction of cauchy decrease constant
       alpha_1     <type:float> line search reduction ratio
       mu          <type:float> termination criteria
       beta        <type:float> termination criteria
       
       theta_1     <type:float> RBF model construction parameter
       theta_2     <type:float> RBF model construction parameter
       theta_3     <type:float> RBF model construction parameter
       theta_4     <type:float> RBF model construction parameter
       
       maxiter     <type:int>   maximum number of iterations
       maxsubiter  <type:int>   maximum number of iterations on TR subproblem
       
       fd_type     <type:int>   finite different type
       LHS         <type:int>   0 for set initial condition
                                NOT IMPEMENTED 1 for latin hypercube


       corr        <type:int>   0 for gaussian
       gammaR      <type:float> gamma parameters
       betaR       <type:array> beta parameters
       optTheta    <type:bool>  True: use maximum likelihood estimator for gamma
                                False: used given gamma
    """
    def __init__(self):

        self.objective = rosenbrock
        self.n = 2
        self.LB = np.array([-2.0, -2.0])
        self.UB = np.array([2.0, 2.0])
        
        self.fid = np.array([1, 2])
        self.pmax = 50
        
        self.eta_0 = 0.0
        self.eta_1 = 0.2
        self.gamma_0 = 0.5
        self.gamma_1 = 2.0
        self.delta_0 = max(1.0, np.linalg.norm(self.UB,np.inf))
        self.delta_max = 1e3*self.delta_0
        
        self.epsilon = 5e-4
        self.kd = 1e-4

        self.alpha_1 = 0.9
        self.mu = 200.0
        self.beta = 3.0
        
        self.theta_1 = 1e-3
        self.theta_2 = 1e-4
        self.theta_3 = 10.0
        self.theta_4 = max(np.sqrt(self.n),10.0)
        
        self.maxiter = 200
        self.maxsubiter = 50
        
        self.fd_type = 0
        self.LHS = 0
        
        self.corr = 0
        self.gammaR = np.array([2.0])
        self.betaR = np.array([])
        self.params = np.concatenate([self.gammaR, self.betaR])
        self.optTheta = False

        self.low_fid_eval = 0
        self.high_fid_eval = 0
        
    def init_models(self):
        if self.LHS == 0:
            point = Point(self.n)
            point.x = np.array([-2.0, 2.0])
            f_low = self.objective(point, self.fid[0])
            f_high = self.objective(point, self.fid[1])
            point.fval = f_high - f_low
            self.low_fid_eval += 1
        elif self.LHS == 1:
            raise AssertionError("Latin hypercube not implemented. Set LHS = 0")
        else:
            raise AssertionError("LHS can be either 1 or 0")

        krig = Kriging(self.corr, self.params, [point])
        print point.fval
        xk = point
        fk = f_high
        f_low = f_low
        mk = f_high
        delta = self.delta_0
        krig, linear = rbfmodel(self.objective, self.fid, xk, [f_low, fk], krig,\
                                    self.theta_1, self.theta_2, self.theta_3, self.theta_4,\
                                    delta, self.delta_max, self.pmax, self.optTheta)
        
if __name__ == "__main__":
    s = Speed()
    s.init_models()
