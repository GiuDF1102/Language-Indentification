import validation as val
import data_utils as du
import dimensionality_reduction as dr
import SVM_classifiers as svm
import math_utils as mu
import numpy as np
import data_visualization as dv

#LOADING DATASET
labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")
featuresZNorm = mu.z_score(features)
C = np.logspace(-3, 5, num=9)

# #LINEAR SVM
# CprimLinearNorm = np.zeros((2, len(C)))
# ##LINEAR SVM NOT NORMALIZED NOT REBALANCED
# minDCFList = np.zeros((2, len(C)))
# for index, pi in enumerate([0.1,0.5]):
#     for cIndex, c in enumerate(C):
#         SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#         _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
#         print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#         minDCFList[index, cIndex] = minDCF
# CprimLinearNorm[0] = minDCFList.mean(axis=0)

# ##LINEAR SVM NORMALIZED NOT REBALANCED
# minDCFList = np.zeros((2, len(C)))
# for index, pi in enumerate([0.1,0.5]):
#     for cIndex, c in enumerate(C):
#         SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#         _, minDCF = val.k_fold(SVMObj, featuresZNorm, labels, 5, (pi, 1, 1))
#         print("Linear SVM Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#         minDCFList[index, cIndex] = minDCF
# CprimLinearNorm[1] = minDCFList.mean(axis=0)

# dv.plotCPrim(C, CprimLinearNorm, ["Linear SVM", "Linear SVM Z-norm"] , "C", "LinearSVM_LinearSVMNorm")

#LINEAR SVM PCA
#CprimLinearNorm = np.zeros((3, len(C)))
##LINEAR SVM PCA NOT NORMALIZED NOT REBALANCED
"""for PCAIndex, nPCA in enumerate([5,4]):
    minDCFList = np.zeros((2, len(C)))
    dataPCA = dr.PCA(features, nPCA)
    for index, pi in enumerate([0.1,0.5]):
        for cIndex, c in enumerate(C):
            SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
            _, minDCF = val.k_fold(SVMObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
            minDCFList[index, cIndex] = minDCF
    CprimLinearNorm[PCAIndex] = minDCFList.mean(axis=0)"""
with open("output_SVM_Poly(2)_minDCFAndCprim.txt", "w") as f:

    for C_ in [0.001,0.01,0.1,1,10,100]:
        for K_ in [0.01,0.1,1,10,100]:
            for c_ in [0.001,0.01,1,10,100]:
                minDCFList=[]
                for pi in [0.1,0.5]:
                    SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=K_, C=C_,d=2)
                    _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
                    print("Poly(2) SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF),file=f)
                    minDCFList.append(minDCF)
                Cprim=np.array(minDCFList).mean(axis=0)
                print("Poly(2) SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim),file=f)
            

    for PCAIndex in [5,4]:
            dataPCA = dr.PCA(features, PCAIndex)
            for C_ in [0.001,0.01,0.1,1,10,100]:
                for K_ in [0.01,0.1,1,10,100]:#usiamo la radice di K come termine di regolarizzazione per kernel non lineari
                    for c_ in [0.001,0.01,1,10,100]:
                        minDCFList=[]
                        for pi in [0.1,0.5]:
                            SVMObj = svm.SVM('Polinomial', balanced=False,c=c_ ,K=K_, C=C_,d=2)
                            _, minDCF = val.k_fold(SVMObj, dataPCA, labels, 5, (pi, 1, 1))
                            print("Poly(2) SVM Not Balanced Not Normalized, minDCF with pi: {} and PCA: {} and C: {} and c:{} and K: {} is {}".format(pi,PCAIndex, C_,c_,K_, minDCF),file=f)
                            minDCFList.append(minDCF)
                        Cprim=np.array(minDCFList).mean(axis=0)
                        print("Poly(2) SVM Not Balanced Not Normalized, Cprim with PCA: {} and C: {} and c:{} and K: {} is {}".format(PCAIndex, C_,c_,K_,Cprim),file=f)

"""
##LINEAR SVM NOT NORMALIZED NOT REBALANCED
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, c in enumerate(C):
        SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
        _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
        print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimLinearNorm[2] = minDCFList.mean(axis=0)

dv.plotCPrim(C, CprimLinearNorm, ["Linear SVM PCA 5", "Linear SVM PCA 4", "Linear SVM no PCA"] , "C", "LinearSVMPCAs")

##################################################

#POLYNOMIAL 2 SVM
CprimPolinomialNorm = np.zeros((2, len(C)))
##POLYNOMIAL SVM NOT NORMALIZED NOT REBALANCED
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, c in enumerate(C):
        SVMObj = svm.SVM('Polinomial', balanced=False, d=2, K=1, C=c, c=0)
        _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 2) SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm[0] = minDCFList.mean(axis=0)

##POLYNOMIAL 2 SVM NORMALIZED NOT REBALANCED
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, c in enumerate(C):
        SVMObj = svm.SVM('Polinomial', balanced=False, d=2, K=1, C=c, c=0)
        _, minDCF = val.k_fold(SVMObj, featuresZNorm, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 2) SVM Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm[1] = minDCFList.mean(axis=0)

dv.plotCPrim(C, CprimPolinomialNorm, ["Polynomial (d = 2) SVM", "Polynomial (d = 2) SVM Z-norm"] , "C", "Plynomial2SVM_Plynomial2SVMNorm")

#POLYNOMIAL 2 SVM PCA
CprimPolinomialNorm = np.zeros((3, len(C)))
##POLYNOMIAL 2 SVM PCA NOT NORMALIZED NOT REBALANCED
for PCAIndex, nPCA in enumerate([5,4]):
    minDCFList = np.zeros((2, len(C)))
    dataPCA = dr.PCA(features, nPCA)
    for index, pi in enumerate([0.1,0.5]):
        for cIndex, c in enumerate(C):
            SVMObj = svm.SVM('Polinomial', balanced=False, d=2, K=1, C=c, c=0)
            _, minDCF = val.k_fold(SVMObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Polynomial (d = 2) SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
            minDCFList[index, cIndex] = minDCF
    CprimPolinomialNorm[PCAIndex] = minDCFList.mean(axis=0)

##POLYNOMIAL 2 SVM NOT NORMALIZED NOT REBALANCED
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, c in enumerate(C):
        SVMObj = svm.SVM('Polinomial', balanced=False, d=2, K=1, C=c, c=0)
        _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 2) SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm[2] = minDCFList.mean(axis=0)

dv.plotCPrim(C, CprimPolinomialNorm, ["Polynomial (d = 2) SVM PCA 5", "Polynomial (d = 2) SVM PCA 4", "Polynomial (d = 2) SVM no PCA"] , "C", "Polynomial2SVMPCAs")
"""