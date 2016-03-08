import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from aux import subset
from libsvm.python import svmutil
import pandas as pd

def eigs(A, k):
    """
    compute eigenvectors corresponding to the
    k largest eigenvalues of A, general version
    ------------------------------------------
    input:
    A --target matrix
    k --num of largest eigenvalues

    output:
    W --projection matrix
    """
    [va, ve] = np.linalg.eig(A)
    st_idx = np.argsort(np.absolute(va))  # sort idx of abs val of eigenvalues
    W = ve[:, np.flipud(st_idx)[:k]]  # sorted eigenvectors, fliped so that small to large
    # info = np.sum(np.flipud(np.sort(np.absolute(va)))[0:k])/np.sum(np.absolute(va))
    # print '%f%% variance reserved' % (info*100)
    return W

def eigs1(A, B, k):
    """
    compute eigenvectors corresponding to the
    k largest eigenvalues of A for B'*B
    ------------------------------------------
    input:
    A --target matrix
    B --sample matrix
    k --num of largest eigenvalues

    output:
    W --projection matrix
    """
    [va, ve] = np.linalg.eig(A)
    dm = A.shape[0]
    st_idx = np.argsort(np.absolute(va))  # sort idx of abs val of eigenvalues
    st_val = np.flipud(np.sort(np.absolute(va)))[:k].reshape(1, k)
    sigma = np.sqrt(np.tile(st_val, (dm, 1)))
    W = B.dot(ve[:, np.flipud(st_idx)[:k]]/sigma)  # sorted eigenvectors, fliped so that small to large
    # info = np.sum(np.flipud(np.sort(np.absolute(va)))[0:k])/np.sum(np.absolute(va))
    # print '%f%% variance reserved' % (info*100)
    return W

def showface(W):
    """
    draw face according to the projection matrix
    --------------------------------------------
    input:
    W --projection mat

    output:
    plot
    """
    fea_dm = W.shape[0]
    F = np.sum(W, axis=-1)
    face = ((F-np.min(F))/(np.max(F)-np.min(F))).reshape((
        np.sqrt(fea_dm), np.sqrt(fea_dm)), order='F').astype(float)
    plt.imshow(face, cmap=plt.get_cmap('gray'))
    plt.show()

def plt_co_mat(co):
    numOlbs = co.shape[0]
    plt.imshow(co, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(numOlbs), np.arange(numOlbs)+1)
    plt.yticks(np.arange(numOlbs), np.arange(numOlbs)+1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def predat(numOtr, cls):
    """
    random subset the dataset, then convert
    to numpy format for further implementations
    -------------------------------------------
    input:
    numOtr --total number of training samples
    cls --number of classes

    output:
    sam_tr, ind_tr --numpy format training samples
    sam_te, ind_te --numpy format testing samples
    cls -- number of classes
    """
    subset.main('dataset/YaleB.scale', numOtr, 0, 'dataset/YaleB.scale.tr', 'dataset/YaleB.scale.te')
    lb_tr, ins_tr = svmutil.svm_read_problem('dataset/YaleB.scale.tr')
    lb_te, ins_te = svmutil.svm_read_problem('dataset/YaleB.scale.te')
    # change training data to numpy format
    df = pd.DataFrame(ins_tr).fillna(0)
    sam_tr = pd.DataFrame.as_matrix(df).T
    ind_tr = np.array(lb_tr)
    # change testing data to numpy format
    df = pd.DataFrame(ins_te).fillna(0)
    sam_te = pd.DataFrame.as_matrix(df).T
    ind_te = np.array(lb_te)
    return sam_tr, ind_tr, sam_te, ind_te, cls

def np2libsvm(ind_tr, sam_tr, ind_te, sam_te):
    """
    convert numpy format data to libsvm format for classification
    ----------------------------------------------
    """
    ins_tr, ins_te = [], []
    for row in sam_tr.T: # 380*1024
        ins_tr.append(dict(enumerate(row)))
    for row in sam_te.T:
        ins_te.append(dict(enumerate(row)))
    lb_tr = list(ind_tr)
    lb_te = list(ind_te)
    return lb_tr, ins_tr, lb_te, ins_te

