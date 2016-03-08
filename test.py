from libsvm.tools import grid
import sklearn.metrics as met
from libsvm.python import svmutil
from aux import subset
import matplotlib.pyplot as plt
import numpy as np
import my_func as mf
from scipy.optimize import minimize

sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*10, 38)
fun = lambda x: np.linalg.norm(x, ord=1)
cons = ({'type': 'eq', 'fun': lambda x: sam_tr.dot(x) - sam_te[:, 1]})
x0 = np.random.randint(10, size=(380, 1))
res = minimize(fun, x0,  method='SLSQP', constraints=cons)
