import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

def predat(num_tr, num_te):
    """Initializing the training set and testing set
    ------------------------------------------------
    inputs:
    num_tr --num of training samples per class
    num_te --num of testing samples per class

    outputs:
    sam_tr --matrix contains training samples
    sam_te --matrix contains testing samples
    num_cls --num of classes
    """
    dataset = sio.loadmat('Yale.mat')
    samples = dataset['fea'].T
    labels = dataset['gnd']
    num_cls = np.unique(labels).size # num of classes
    num_to = num_tr + num_te  # num of total samples in each class
    # indices of samples, left half training, right half testing
    ind = np.array([np.random.permutation(
    np.where(labels==i)[0][0:num_to]) for i in range(1, num_cls+1)])
    training_idx, testing_idx = ind[:, 0:num_tr].flatten(0), \
                            ind[:, num_tr:num_to].flatten(0)
    # training set and testing set
    sam_tr, sam_te = samples[:, training_idx], samples[:, testing_idx]
    return sam_tr, sam_te, num_cls

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
