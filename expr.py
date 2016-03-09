import mthds as md
import numpy as np
import my_func as mf
from aux import subset

# initialization
expr_time = 1
tr = 29
expr = [1, 2, 3, 4, 5]
"""
0. showface
1. fisherface influence of k
2. fisherface influence of training set size
3. eigenface influence of k
4. eigenface influence of training set size
5. svm influence of training set size
"""

# fisherface
# expr1, show fisherface
for numOexpr in expr:
    if numOexpr == 0:
        k = 100
        sam_tr, sam_te, c = mf.predat(tr, te)
        a, co, w = md.lda(k, sam_tr, sam_te, c)
        mf.showface(w)

    # expr2, influence of k
    if numOexpr == 1:
        infOk = np.zeros(7)
        idx = 0
        for k in np.array([10, 15, 20, 25, 30, 35, 40]):
            ac = 0
            for i in range(1, expr_time+1):
                # sam_tr, sam_te, c = mf.predat(tr, te)
                sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*50, 38)
                a, co, w = md.lda(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=0)
                ac += a
            accu = ac/expr_time
            infOk[idx] = accu
            idx += 1
        # print "accuracy is %0.2f%%" % (accu*100)
        np.save('results/inflk_fisherface', infOk)

    # expr3, influence of training set size
    if numOexpr == 2:
        infOtr = np.zeros(5)
        idx = 0
        for tr in np.array([10, 20, 30, 40, 50]):
            ac = 0
            for i in range(1, expr_time+1):
                # sam_tr, sam_te, c = mf.predat(tr, te)
                sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*tr, 38)
                a, co, w = md.lda(25, sam_tr, ind_tr, sam_te, ind_te, c, classifier=0)
                ac += a
            accu = ac/expr_time
            infOtr[idx] = accu
            idx += 1
        # print "accuracy is %0.2f%%" % (accu*100)
        np.save('results/infltr_fisherface', infOk)

    # eigenface
    # expr1, influence of k
    if numOexpr == 3:
        infOk = np.zeros(7)
        idx = 0
        for k in np.array([50, 100, 150, 200, 250, 300, 500]):
            ac = 0
            for i in range(1, expr_time+1):
                sam_tr, ind_tr, sam_te, ind_te, c  = mf.predat(38*50, 38)
                a, co, w = md.pca(k, sam_tr, ind_tr, sam_te, ind_te, c, classifier=1)
                ac += a
            accu = ac/expr_time
            infOk[idx] = accu
            idx += 1
        # print "accuracy is %0.2f%%" % (accu*100)
        np.save('results/inflk_eigenface', infOk)

    # expr2, influence of training set size
    if numOexpr == 4:
        infOtr = np.zeros(5)
        idx = 0
        for tr in np.array([10, 20, 30, 40, 50]):
            ac = 0
            for i in range(1, expr_time+1):
                sam_tr, ind_tr, sam_te, ind_te, c  = mf.predat(38*tr, 38)
                a, co, w = md.pca(100, sam_tr, ind_tr, sam_te, ind_te, c, classifier=0)
                ac += a
            accu = ac/expr_time
            infOtr[idx] = accu
            idx += 1
        # print "accuracy is %0.2f%%" % (accu*100)
        np.save('results/infltr_eigenface', infOtr)

    # SVM
    # expr1, raw svm
    if numOexpr == 5:
        infOtr = np.zeros(5)
        idx = 0
        for tr in np.array([10, 20,30, 40, 50]):
            ac = 0
            for i in range(1, expr_time+1):
                sam_tr, ind_tr, sam_te, ind_te, c = mf.predat(38*tr, 38)
                lb_tr, ins_tr, lb_te, ins_te = mf.np2libsvm(ind_tr, sam_tr, ind_te, sam_te)
                a, co_mat = md.libsvm(lb_tr, ins_tr, lb_te, ins_te)
                ac += a
            accu = ac/expr_time
            infOtr[idx] = accu
            idx += 1
        # print "accuracy is %0.2f%%" % (accu)
        np.save('results/infltr_svm', infOtr)
