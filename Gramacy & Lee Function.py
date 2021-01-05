import numpy as np
import pandas as pd 
from pyDOE import *     # for Latin Hypercue Design
import random
import scipy
from scipy import stats
import GPy
import sklearn
from sklearn import gaussian_process

#### GRAMACY & LEE (2008) FUNCTION (d = 2 - exp/log)
def grlee08(x):
    x1 = x[:,0]
    x2 = x[:,1]  
    fact1 = x1
    fact2 = np.exp(-x1**2 - x2**2)
    y = fact1 * fact2
    return(y)

rmse_grlee = pd.DataFrame(columns = ["gree08_n20", "grlee08_n40"],
                        index=range(100))

for j in range(2):
    rmse_arr = [0]*100
    n = [20, 40]
    random.seed(100)
    x_train = lhs(2, samples = n[j])
    x_train[:,0] = stats.uniform(loc = -2, scale = 6-(-2)).ppf(x_train[:,0])
    x_train[:,1] = stats.uniform(loc = -2, scale = 6-(-2)).ppf(x_train[:,1])
    y_train = np.reshape(grlee08(x_train), (n[j], 1))
    for i in range(100):
        x_test = lhs(2, samples = 1000)
        x_test[:,0] = stats.uniform(loc = -2, scale = 6-(-2)).ppf(x_test[:,0])
        x_test[:,1] = stats.uniform(loc = -2, scale = 6-(-2)).ppf(x_test[:,1])
        y_test = np.reshape(grlee08(x_test), (1000, 1))       
        
        ## GPy               Uncomment to check 
# =============================================================================
#         k_gpy = GPy.kern.Matern52(input_dim = 2, lengthscale = 1)
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
       
    rmse_grlee.iloc[:, j] = np.reshape(rmse_arr, (100, 1))
