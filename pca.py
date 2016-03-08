import numpy as np
import my_func as mf
import matplotlib.pyplot as plt

num_tr = 29
num_te = 1
k = 100
# training
sam_tr, sam_te, dm, c = mf.predat(num_tr, num_te)
mu = np.mean(sam_tr, axis=1)  # compute overall mean
mu_c = np.mean(sam_tr.T.reshape(c, num_tr, dm), axis=1).T  # contain all class centres
B = sam_tr - np.tile(mu.reshape(dm, 1), (1, num_tr*c))
S = B.dot(B.T)
# S = sam_tr.dot(sam_tr.T)
W = mf.eigs(S, k)  # the projection matrix
mf.showface(W)
# testing
sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, c*num_te)))  # projected testing samples
mu_c_p = W.T.dot(mu_c-np.tile(mu.reshape(dm, 1), (1, c)))  # projected class means
# 1NN
sam_te_p_l = np.tile(sam_te_p.T, (c, 1, 1)).transpose(1, 0, 2) \
    .reshape(num_te*c*c, k).T  # every c row is one sample
mu_c_p_l = np.tile(mu_c_p, (1, c*num_te))  # every sample versus all centres
res_1nn = np.linalg.norm(sam_te_p_l-mu_c_p_l, axis=0).reshape(num_te*c, c)
# row: samples, col: classes, all below res mats are similar
res_min = np.tile(np.min(res_1nn, axis=1).reshape(num_te*c, 1), (1, c))
res_bin = res_1nn == res_min  # boolean data
co_mat = np.sum(res_bin.reshape(c, num_te, c), axis=1)
# confusion mat, use float for possible fuzzy improvement
accu = np.trace(co_mat).astype(float)/np.sum(co_mat).astype(float)
print accu
