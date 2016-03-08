import mthds as md
import numpy as np
import my_func as mf
from aux import subset

# initialization
expr_time = 1
tr = 29
ac = 0
numOexpr = 2

# fisherface
# expr1, show fisherface
if numOexpr == 1:
    k = 100
    sam_tr, sam_te, c = mf.predat(tr, te)
    a, co, w = md.lda(k, sam_tr, sam_te, c)
    mf.showface(w)

# expr2, influence of k
if numOexpr == 2:
    infOk = np.zeros([1, 7])
    idx = 0
    for k in np.array([10, 15, 20, 25, 30, 35, 40]):
        for i in range(1, expr_time+1):
            # sam_tr, sam_te, c = mf.predat(tr, te)
            sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*50, 38)
            a, co, w = md.lda(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=1)
            ac += a
        accu = ac/expr_time
        infOk[idx] = accu
        idx += 1
    # print "accuracy is %0.2f%%" % (accu*100)
    np.save('results/inflk_fisherface', infOk)

# eigenface
# expr1, influence of k
if numOexpr == 3:
    infOk = np.zeros([1, 7])
    idx = 0
    for k in np.array([50, 100, 150, 200, 250, 300, 500]):
        for i in range(1, expr_time+1):
            sam_tr, ind_tr, sam_te, ind_te, c  = mf.predat(38*tr, 38)
            a, co, w = md.pca(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=1)
            ac += a
        accu = ac/expr_time
        infOk[idx] = accu
        idx += 1
    # print "accuracy is %0.2f%%" % (accu*100)
    np.save('results/inflk_eigenface', infOk)

# SVM
# expr1, raw svm
if numOexpr == 4:
    for i in range(1, expr_time+1):
        sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*tr, 38)
        lb_tr, ins_tr, lb_te, ins_te = mf.np2libsvm(ind_tr, sam_tr, ind_te, sam_te)
        a, co_mat = md.libsvm(lb_tr, ins_tr, lb_te, ins_te)
        ac += a
    accu = ac/expr_time
    print "accuracy is %0.2f%%" % (accu)
