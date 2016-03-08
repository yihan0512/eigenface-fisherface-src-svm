from libsvm.tools import grid
import sklearn.metrics as met
from libsvm.python import svmutil
from aux import subset
import matplotlib.pyplot as plt
import numpy as np
import my_func as mf
subset.main('dataset/YaleB.scale', 380, 0, 'dataset/YaleB.scale.tr', 'dataset/YaleB.scale.te')

rate, param = grid.find_parameters('dataset/YaleB.scale.tr', '-log2c -1,1,1 -log2g -1,1,1')
lb_tr, ins_tr = svmutil.svm_read_problem('dataset/YaleB.scale.tr')
lb_te, ins_te = svmutil.svm_read_problem('dataset/YaleB.scale.te')
prob  = svmutil.svm_problem(lb_tr, ins_tr)
param = svmutil.svm_parameter('-c %f -g %f' % (param['c'], param['g']))
m = svmutil.svm_train(prob, param)
p_label, p_acc, p_val = svmutil.svm_predict(lb_te, ins_te, m)
print p_acc
co = met.confusion_matrix(lb_te, p_label, labels=np.arange(5)+1)
mf.plt_co_mat(co)
