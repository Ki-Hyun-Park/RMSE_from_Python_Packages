import numpy as np
import pandas as pd 
from pyDOE import *     # for Latin Hypercue Design
import random
import scipy
from scipy import stats
import GPy
import sklearn
from sklearn import gaussian_process
#### Generate data x and y for Borehold function ####

#### BOREHOLE FUNCTION (d = 8 - physical) (pending)

def borehole(x):
    rw = x[:,0]
    r = x[:,1]
    tu = x[:,2]
    hu = x[:,3]
    tl = x[:,4]
    hl = x[:,5]
    l = x[:,6]
    kw = x[:,7]
    frac1 = 2 * np.pi * tu * (hu-hl)
    frac2a = 2*l*tu / (np.log(r/rw)*rw**2*kw)
    frac2b = tu / tl
    frac2 = np.log(r/rw) * (1+frac2a+frac2b)
    y = frac1 / frac2
    return(y)

random.seed(100)
x_train = lhs(8, samples = 80)
x_train[:, 0] = stats.uniform(loc = 0.05, scale = 0.15-0.05).ppf(x_train[:, 0])
x_train[:, 1] = stats.uniform(loc = 100, scale = 50000-100).ppf(x_train[:, 1])
x_train[:, 2] = stats.uniform(loc = 63070, scale = 115600-63070).ppf(x_train[:, 2])
x_train[:, 3] = stats.uniform(loc = 990, scale = 1110-990).ppf(x_train[:, 3])
x_train[:, 4] = stats.uniform(loc = 63.1, scale = 116-63.1).ppf(x_train[:, 4])
x_train[:, 5] = stats.uniform(loc = 700, scale = 820-700).ppf(x_train[:, 5])
x_train[:, 6] = stats.uniform(loc = 1120, scale = 1680-1120).ppf(x_train[:, 6])
x_train[:, 7] = stats.uniform(loc = 9855, scale = 12045-9855).ppf(x_train[:, 7])
y_train = np.reshape(borehole(x_train), (80,1))

random.seed(1000)
x_test = lhs(8, samples = 1000)
x_test[:,0] = stats.uniform(loc = 0.05, scale = 0.15-0.05).ppf(x_test[:, 0])
x_test[:, 1] = stats.uniform(loc = 100, scale = 50000-100).ppf(x_test[:, 1])
x_test[:,2] = stats.uniform(loc = 63070, scale = 115600-63070).ppf(x_test[:, 2])
x_test[:,3] = stats.uniform(loc = 990, scale = 1110-990).ppf(x_test[:, 3])
x_test[:, 4] = stats.uniform(loc = 63.1, scale = 116-63.1).ppf(x_test[:, 4])
x_test[:, 5] = stats.uniform(loc = 700, scale = 820-700).ppf(x_test[:, 5])
x_test[:, 6] = stats.uniform(loc = 1120, scale = 1680-1120).ppf(x_test[:, 6])
x_test[:, 7] = stats.uniform(loc = 9855, scale = 12045-9855).ppf(x_test[:, 7])
y_test = np.reshape(borehole(x_test), (1000,1))


## GPy
random.seed(99999)
k_gpy = GPy.kern.Matern52(input_dim = 8, lengthscale = 1)
m_gpy = GPy.models.GPRegression(x_train, y_train, k_gpy, noise_var = 10**(-8))
m_gpy.optimize('bfgs')
print(m_gpy)
y_pred, var_pred = m_gpy.predict(x_test)

r = m_gpy.predict(x_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
np.min(y_pred)
np.max(y_pred)

## sklearn
random.seed(100)
k = gaussian_process.kernels.Matern(length_scale=1.0)
sk_gp = gaussian_process.GaussianProcessRegressor(kernel = k, optimizer='fmin_l_bfgs_b')
sk_fit = sk_gp.fit(x_train, y_train)
print(sk_fit)
y_pred = sk_fit.predict(x_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))


#################### CHECK/RUN FROM HERE #######################################
# NOTE: For all functions, only run one package first, export the results 
        # then run another package with similar steps 
