import numpy as np
import my_func as mf
import scipy.stats as ss
from libsvm.tools import grid
from libsvm.python import svmutil
import sklearn.metrics as met
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn import svm


def lda(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=0):
    """
    fisherface implementation
    -------------------------
    inputs:
    k --size of subspace;
    sam_tr --training set
    sam_te --testing set
    c --num of classes
    classifier --0 for kNN(default), 1 for SVM

    outputs:
    accu --classification accuracy
    co_mat --confusion mat
    W --projection mat
    """
    # data preparation
    flg = 0
    num_tr = sam_tr.shape[1]/c
    dm = sam_tr.shape[0]

    # training
    # compute variables
    # compute within class mean
    mu_c = np.mean(sam_tr.T.reshape(c, num_tr, dm), axis=1).T
    mu = np.mean(sam_tr, 1)  # compute overall mean
    # compute within class scatter matrix Sw
    mu_c_l = np.tile(mu_c.T, (num_tr, 1, 1)).transpose(1, 0, 2)\
        .reshape(num_tr*c, dm).T  # enlarged mean mat, note the usage of transpose
    B = sam_tr - mu_c_l
    Sw = B.dot(B.T)
    # compute between class scatter matrix Sb
    Ni = num_tr*np.ones((dm, c))  # sample numbers in each class
    C = mu_c - np.tile(mu.reshape(dm, 1), (1, c))
    Sb = C.dot(Ni.T*C.T)
    # solve generalized eigenvalue problem
    W = mf.eigs(np.linalg.inv(Sw).dot(Sb), k)  # the projection matrix

    # testing
    sam_tr_p = W.T.dot(sam_tr-np.tile(mu.reshape(dm, 1), (1, sam_tr.shape[1])))  # projected training samples
    sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, sam_te.shape[1])))  # projected testing samples
    # 1NN
    if classifier == 0:
        nei = kNN(n_neighbors=1)
        nei.fit(sam_tr_p.T, ind_tr)
        p_label = nei.predict(sam_te_p.T)
        co_mat = met.confusion_matrix(ind_te, p_label)
    # SVM
    if classifier == 1:
        lb_tr, ins_tr, lb_te, ins_te = mf.np2libsvm(ind_tr, sam_tr_p, ind_te, sam_te_p)
        accu, co_mat = libsvm(lb_tr, ins_tr, lb_te, ins_te)
        flg = 1

    if flg == 0:
        accu = np.trace(co_mat).astype(float)/np.sum(co_mat).astype(float)
    else:
        accu = accu/100
    return accu, co_mat, W

def pca(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=0):
    """
    eigenface implementation
    ----------------------------------------
    inputs:
    k --size of subspace
    sam_tr --training set
    sam_te --testing set
    c --num of classes
    classifier --0 for kNN(default), 1 for SVM

    outputs
    accu --classfication accuracy
    co_mat --confusion mat
    W --projection mat
    """
    # data preparation
    flg = 0
    num_tr = sam_tr.shape[1]/c
    dm = sam_tr.shape[0]

    # training
    mu = np.mean(sam_tr, axis=1)  # compute overall mean
    B = sam_tr - np.tile(mu.reshape(dm, 1), (1, num_tr*c))
    S = B.T.dot(B)
    # S = sam_tr.dot(sam_tr.T)
    W = mf.eigs1(S, B, k)  # the projection matrix
    W = np.real(W)

    # testing
    sam_tr_p = W.T.dot(sam_tr-np.tile(mu.reshape(dm, 1), (1, sam_tr.shape[1])))  # projected training samples
    sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, sam_te.shape[1])))  # projected testing samples
    # 1NN
    if classifier == 0:
        nei = kNN(n_neighbors=1)
        nei.fit(sam_tr_p.T, ind_tr)
        p_label = nei.predict(sam_te_p.T)
        co_mat = met.confusion_matrix(ind_te, p_label)
    # SVM
    if classifier == 1:
        lb_tr, ins_tr, lb_te, ins_te = mf.np2libsvm(ind_tr, sam_tr_p, ind_te, sam_te_p)
        accu, co_mat = libsvm(lb_tr, ins_tr, lb_te, ins_te)
        flg = 1

    if flg == 0:
        accu = np.trace(co_mat).astype(float)/np.sum(co_mat).astype(float)
    else:
        accu = accu/100
    return accu, co_mat, W

def libsvm(lb_tr, ins_tr, lb_te, ins_te):
    """
    libsvm classifier, using libsvm
    -------------------------------
    read normalized libsvm format files from ./dataset

    output:
    accu --classification accuracy
    co --confusion matrix
    """
    rate, param = grid.find_parameters('dataset/YaleB.scale.tr', '-log2c -1,1,1 -log2g -1,1,1')
    prob  = svmutil.svm_problem(lb_tr, ins_tr)
    param = svmutil.svm_parameter('-c %f -g %f' % (param['c'], param['g']))
    m = svmutil.svm_train(prob, param)
    p_label, p_acc, p_val = svmutil.svm_predict(lb_te, ins_te, m)
    co = met.confusion_matrix(lb_te, p_label)
    accu = p_acc[0]
    return accu, co

