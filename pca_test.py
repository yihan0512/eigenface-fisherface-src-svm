import numpy as np
import scipy.io as sio
import my_func as mf
import matplotlib.pyplot as plt

dataset = sio.loadmat('Yale.mat')
samples = dataset['fea'].T
labels = dataset['gnd']
num_cls = np.unique(labels).size # num of classes

B = samples - np.tile(np.mean(samples, 1).reshape(1024, 1), (1, 2414))
S = B.dot(B.T)
W = mf.eigs(S, 500)
mf.showface(W)
# mf.showface(samples[:, 1].reshape(1024, 1))
# plt.imshow(samples[:, 1].reshape(32,32))

