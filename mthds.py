import numpy as np
import my_func as mf
import scipy.stats as ss


def lda(k, sam_tr, sam_te, c):
    """
    fisherface implementation
    -------------------------
    inputs:
    k --size of subspace;
    sam_tr --training set
    sam_te --testing set
    c --num of classes

    outputs:
    accu --classification accuracy
    co_mat --confusion mat
    W --projection mat
    """
    # data preparation
    num_tr = sam_tr.shape[1]/c
    num_te = sam_te.shape[1]/c
    dm = sam_tr.shape[0]

    # training
    # compute variables
    # compute within class mean
    # a, co, Wpca = pca(num_tr*c-c, sam_tr, sam_te, c)
    # sam_tr = Wpca.T.dot(sam_tr)
    # sam_te = Wpca.T.dot(sam_te)
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
    # W = Wpca.dot(Wlda)

    # testing
    # 1NN using class centres
    if 0:
        sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, c*num_te)))  # projected testing samples
        mu_c_p = W.T.dot(mu_c-np.tile(mu.reshape(dm, 1), (1, c)))  # projected class means
        # 1NN
        sam_te_p_l = np.tile(sam_te_p.T, (c, 1, 1)).transpose(1, 0, 2)\
            .reshape(num_te*c*c, k).T  # every c row is one sample
        mu_c_p_l = np.tile(mu_c_p, (1, c*num_te))  # every sample versus all centres
        res_1nn = np.linalg.norm(sam_te_p_l-mu_c_p_l, axis=0).reshape(num_te*c, c)
                                            # row: samples, col: classes, all below res mats are similar
        res_min = np.tile(np.min(res_1nn, axis=1).reshape(num_te*c, 1), (1, c))
        res_bin = res_1nn == res_min  # boolean data
        co_mat = np.sum(res_bin.reshape(c, num_te, c), axis=1)
            # confusion mat, use float for possible fuzzy improvement

    # 1NN(possible kNN) using all training samples
    if 1:
        sam_tr_p = W.T.dot(sam_tr-np.tile(mu.reshape(dm, 1), (1, c*num_tr)))  # projected training samples
        sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, c*num_te)))  # projected testing samples
        co_mat = np.zeros((c, c))
        ind_tr = np.tile(np.arange(0, c).reshape(c, 1), (1, num_tr)).flatten(0)
        ind_te = np.tile(np.arange(0, c).reshape(c, 1), (1, num_te)).flatten(0)
        for i in range(0, c*num_te):
            tmp = np.tile(sam_te_p[:, i].reshape(k, 1), (1, c*num_tr))
            dist = np.linalg.norm(tmp-sam_tr_p, axis=0)
            ind = np.argsort(dist)
            j = ss.mode(ind_tr[ind[:20]])
            co_mat[ind_te[i], j[0]] += 1

    accu = np.trace(co_mat).astype(float)/np.sum(co_mat).astype(float)
    return accu, co_mat, W

def pca(k, sam_tr, sam_te, c):
    """
    eigenface implementation
    ----------------------------------------
    inputs:
    k --size of subspace
    sam_tr --training set
    sam_te --testing set
    c --num of classes

    outputs
    accu --classfication accuracy
    co_mat --confusion mat
    W --projection mat
    """
    # data preparation
    num_tr = sam_tr.shape[1]/c
    num_te = sam_te.shape[1]/c
    dm = sam_tr.shape[0]

    # training
    mu = np.mean(sam_tr, axis=1)  # compute overall mean
    mu_c = np.mean(sam_tr.T.reshape(c, num_tr, dm), axis=1).T  # contain all class centres
    B = sam_tr - np.tile(mu.reshape(dm, 1), (1, num_tr*c))
    S = B.T.dot(B)
    # S = sam_tr.dot(sam_tr.T)
    W = mf.eigs1(S, B, k)  # the projection matrix

    # testing
    # 1NN, all samples, matrix manipulation, too slow
    if 0:
        sam_tr_p = W.T.dot(sam_tr-np.tile(mu.reshape(dm, 1), (1, c*num_tr)))  # projected training samples
        sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, c*num_te)))  # projected testing samples
        # 1NN
        sam_te_p_l = np.tile(sam_te_p.T, (c*num_tr, 1, 1)).transpose(1, 0, 2) \
            .reshape(num_te*c*num_tr*c, k).T  # every c row is one sample
        sam_tr_p_l = np.tile(sam_tr_p, (1, c*num_te))  # every sample versus all centres
        res_1nn = np.linalg.norm(sam_te_p_l-sam_tr_p_l, 2, axis=0).reshape(c*num_te, c*num_tr)
                # row: samples, col: classes, all below res mats are similar
        res_min = np.tile(np.min(res_1nn, axis=1).reshape(num_te*c, 1), (1, num_tr*c))
        res_bin = res_1nn == res_min  # boolean data
        co_mat = np.sum(np.sum(res_bin.reshape(num_te*c, c, num_tr), axis=2).reshape(c, num_te, c), axis=1)
                # confusion mat, use float for possible fuzzy improvement

    # 1NN(possible kNN), all samples, loop, faster
    if 1:
        sam_tr_p = W.T.dot(sam_tr-np.tile(mu.reshape(dm, 1), (1, c*num_tr)))  # projected training samples
        sam_te_p = W.T.dot(sam_te-np.tile(mu.reshape(dm, 1), (1, c*num_te)))  # projected testing samples
        co_mat = np.zeros((c, c))
        ind_tr = np.tile(np.arange(0, c).reshape(c, 1), (1, num_tr)).flatten(0)
        ind_te = np.tile(np.arange(0, c).reshape(c, 1), (1, num_te)).flatten(0)
        for i in range(0, c*num_te):
            tmp = np.tile(sam_te_p[:, i].reshape(k, 1), (1, c*num_tr))
            dist = np.linalg.norm(tmp-sam_tr_p, axis=0)
            ind = np.argsort(dist)
            j = ss.mode(ind_tr[ind[:1]])
            co_mat[ind_te[i], j[0]] += 1

    accu = np.trace(co_mat).astype(float)/np.sum(co_mat).astype(float)
    return accu, co_mat, W
