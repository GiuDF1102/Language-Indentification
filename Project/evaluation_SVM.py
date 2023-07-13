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
CprimLinearNorm = np.zeros((3, len(C)))
##LINEAR SVM PCA NOT NORMALIZED NOT REBALANCED
for PCAIndex, nPCA in enumerate([5,4]):
    minDCFList = np.zeros((2, len(C)))
    dataPCA = dr.PCA(features, nPCA)
    for index, pi in enumerate([0.1,0.5]):
        for cIndex, c in enumerate(C):
            SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
            _, minDCF = val.k_fold(SVMObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
            minDCFList[index, cIndex] = minDCF
    CprimLinearNorm[PCAIndex] = minDCFList.mean(axis=0)

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