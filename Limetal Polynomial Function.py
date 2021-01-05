import numpy as np
import pandas as pd 
from pyDOE import *     # for Latin Hypercue Design
import random
import scipy
from scipy import stats
import GPy
import sklearn
from sklearn import gaussian_process

def limetal02pol(xx):
    x1 = xx[:, 0]
    x2 = xx[:, 1]
    term1 = (5/2)*x1 - (35/2)*x2
    term2 = (5/2)*x1*x2 + 19*x2**2
    term3 = -(15/2)*x1**3 - (5/2)*x1*x2**2
    term4 = -(11/2)*x2**4 + (x1**3)*(x2**2)
    y = 9 + term1 + term2 + term3 + term4
    return(y)
    
rmse_lim = pd.DataFrame(columns = ["limetal02pol_n20", "limetal02pol_n40"],
                        index=range(100))
for j in range(2):           
    rmse_arr = [0]*100
    n = [20, 40]
    random.seed(100)
    x_train = lhs(2, samples = n[j])
    y_train = np.reshape(limetal02pol(x_train), (n[j], 1))
    for i in range(100):
        x_test = lhs(2, samples = 1000)
        y_test = np.reshape(limetal02pol(x_test), (1000, 1))
        
        ## GPy        Uncomment to run only after Sklearn has been done and exported
# =============================================================================
#         k_gpy = GPy.kern.Matern52(input_dim = x_train.shape[1], lengthscale = 1)
#         m_gpy = GPy.models.GPRegression(x_train, y_train, k_gpy, noise_var = 10**(-8))
#         m_gpy.optimize('bfgs')
#         y_pred, var_pred = m_gpy.predict(x_test)
#         rmse_arr[i] = np.sqrt(np.mean((y_test - y_pred)**2))
# =============================================================================

        ## sklearn
        k = gaussian_process.kernels.Matern(length_scale=1.0)
        sk_gp = gaussian_process.GaussianProcessRegressor(kernel = k, optimizer='fmin_l_bfgs_b')
        sk_fit = sk_gp.fit(x_train, y_train)
        y_pred = sk_fit.predict(x_test)
        rmse_arr[i] = np.sqrt(np.mean((y_test - y_pred)**2))
        
    rmse_lim.iloc[:, j] = np.reshape(rmse_arr, (100, 1))
