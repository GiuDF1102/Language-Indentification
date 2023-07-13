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
C = np.logspace(-3, 5, num=9)
CprimLinearNoPCAK = np.zeros((3, len(C)))

##LINEAR SVM No PCA
features = dr.PCA(features, 5)
for kIndex,k in enumerate([0.01,0.1,1]):
    minDCFList = np.zeros((2, len(C)))
    for index, pi in enumerate([0.1,0.5]):
        for cIndex, c in enumerate(C):
            SVMObj = svm.SVM('linear', balanced=False, K=k, C=c)
            _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
            print(f"Linear SVM PCA 5, minDCF with pi {pi}, K {k} and C {c} is {minDCF}")
            minDCFList[index, cIndex] = minDCF
    CprimLinearNoPCAK[kIndex] = minDCFList.mean(axis=0)

dv.plotCPrim(C, CprimLinearNoPCAK, ["Linear SVM PCA 5 K = 0.01", "Linear SVM PCA 5 K = 0.1", "Linear SVM PCA 5 K = 1"] , "C", "LinearSVMPCA5Ks")