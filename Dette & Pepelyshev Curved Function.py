import numpy as np
import pandas as pd 
from pyDOE import *     # for Latin Hypercue Design
import random
import scipy
from scipy import stats
import GPy
import sklearn
from sklearn import gaussian_process

#### DETTE & PEPELYSHEV (2010) CURVED FUNCTION (3d)
def detpep10curv(x):
  x1 = x[:, 0]
  x2 = x[:, 1]
  x3 = x[:, 2]
  term1 = 4 * (x1 - 2 + 8*x2 - 8*x2**2)**2
  term2 = (3 - 4*x2)**2
  term3 = 16 * np.sqrt(x3+1) * (2*x3-1)**2
  y = term1 + term2 + term3
  return(y)

rmse_detpep = pd.DataFrame(columns = ["detpep10curv_n30", "detpep10curv_n60"],
                        index=range(100))


for j in range(2):
    rmse_arr = [0]*100
    n = [30, 60]
    random.seed(100)
    x_train = lhs(3, samples = n[j])
    y_train = np.reshape(detpep10curv(x_train), (n[j], 1))
    for i in range(100):
        x_test = lhs(3, samples = 1000)
        y_test = np.reshape(detpep10curv(x_test), (1000, 1))
        
        ## GPy             Uncomment to check 
# =============================================================================
#         k_gpy = GPy.kern.Matern52(input_dim = 3, lengthscale = 1)
#         m_gpy = GPy.models.GPRegression(x_train, y_train, k_gpy, noise_var = 10**(-8))
#         m_gpy.optimize('bfgs')
#         y_pred, var_pred = m_gpy.predict(x_test)
#         rmse_arr[i] = np.sqrt(np.mean((y_test - y_pred)**2))
# 
# =============================================================================
        ## sklearn
        k = gaussian_process.kernels.Matern(length_scale=1.0)
        sk_gp = gaussian_process.GaussianProcessRegressor(kernel = k, optimizer='fmin_l_bfgs_b')
        sk_fit = sk_gp.fit(x_train, y_train)
        y_pred = sk_fit.predict(x_test)
        rmse_arr[i] = np.sqrt(np.mean((y_test - y_pred)**2))
       
    rmse_detpep.iloc[:, j] = np.reshape(rmse_arr, (100, 1))
