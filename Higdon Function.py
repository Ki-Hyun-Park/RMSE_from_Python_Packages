import numpy as np
import pandas as pd 
from pyDOE import *     # for Latin Hypercue Design
import random
import scipy
from scipy import stats
import GPy
import sklearn
from sklearn import gaussian_process

#### HIGDON (2002) FUNCTION (1d - Trigonometry)
def hig02(s):
    term1 = np.sin(2*np.pi*s/10)
    term2 = 0.2 * np.sin(2*np.pi*s/2.5)
    y = term1 + term2
    return(y)
    
    
rmse_hig = pd.DataFrame(columns = ["hig02_n10", "hig02_n20"],
                        index=range(100))

for j in range(2):
    rmse_arr = [0]*100
    n = [10, 20]
    random.seed(100)
    x_train = lhs(1, samples = n[j])
    x_train = stats.uniform(loc = 0, scale = 10-0).ppf(x_train)
    y_train = np.reshape(hig02(x_train), (n[j], 1))
    for i in range(100):
        x_test = lhs(1, samples = 1000)
        x_test = stats.uniform(loc = 0, scale = 10-0).ppf(x_test)
        y_test = np.reshape(hig02(x_test), (1000, 1))
        
        ## GPy            Uncomment to check 
# =============================================================================
#         k_gpy = GPy.kern.Matern52(input_dim = 1, lengthscale = 1)
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
       
    rmse_hig.iloc[:, j] = np.reshape(rmse_arr, (100, 1))


column_names = ["Package", "limetal02pol_n20", "limetal02pol_n40",	
                "detpep10curv_n30",	"detpep10curv_n60",
                "gree08_n20", "grlee08_n40", "hig02_n10", "hig02_n20"]
pack = pd.DataFrame(np.reshape(np.repeat("Sklearn", 100), (100,1)))
# pack = pd.DataFrame(np.reshape(np.repeat("GPy", 100), (100,1)))  # uncomment for GPy
rmse = pd.concat([pack, rmse_lim, rmse_detpep, rmse_grlee, rmse_hig], axis=1, ignore_index=True)
rmse.columns=column_names
export_csv = rmse.to_csv (r'C:\Users\cliccuser\Downloads\rmse_sklearn.csv', index = None, header=True)
# export_csv = rmse.to_csv (r'C:\Users\cliccuser\Downloads\rmse_gpy.csv', index = None, header=True) # uncomment for GPy
