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
namesList = [
    "Polinomial(2) SVM c = 0.1",
    "Polinomial(2) SVM c = 1",
    "Polinomial(2) SVM c = 10"    
]

dataPCA5 = dr.PCA(features, 5)
dataPCA4 = dr.PCA(features, 4)
k = 5

# with open("output_SVM_Polinomial2_minDCFAndCprim.txt", "w") as f:

#     for K_ in [1,10,100]:
#         Cprimlists = []
#         for c_ in [0.1,1,10]:
#             CprimlistForGamma = []
#             for C_ in [0.001,0.01,0.1,1,10,100]:
#                 minDCFList=[]
#                 for pi in [0.1,0.5]:
#                     SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=K_, C=C_, d=2)
#                     _, minDCF = val.k_fold(SVMObj, features, labels, k, (pi, 1, 1))
#                     print("Polinomial SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF),file=f)
#                     print("Polinomial SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF))
#                     minDCFList.append(minDCF)
#                 Cprim=np.array(minDCFList).mean(axis=0)
#                 print("Polinomial SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim),file=f)
#                 print("Polinomial SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim))
#                 CprimlistForGamma.append(Cprim)
#                 print(CprimlistForGamma)
#             if c_ in [0.1,1,10]:
#                 Cprimlists.append(CprimlistForGamma)
#         dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"Polinomial2NoPCA K = {K_}")

#     for K_ in [1,10,100]:
#         Cprimlists = []
#         for c_ in [0.1,1,10]:
#             CprimlistForGamma = []
#             for C_ in [0.001,0.01,0.1,1,10,100]:
#                 minDCFList=[]
#                 for pi in [0.1,0.5]:
#                     SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=K_, C=C_, d=2)
#                     _, minDCF = val.k_fold(SVMObj, dataPCA5, labels, k, (pi, 1, 1))
#                     print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF),file=f)
#                     print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF))
#                     minDCFList.append(minDCF)
#                 Cprim=np.array(minDCFList).mean(axis=0)
#                 print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim),file=f)
#                 print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim))
#                 CprimlistForGamma.append(Cprim)
#                 print(CprimlistForGamma)
#             if c_ in [0.1,1,10]:
#                 Cprimlists.append(CprimlistForGamma)
#         dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"Polinomial2PCA5 K = {K_}")


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

#Test polinomial 3
C = [0.001,0.01,0.1,1]
CprimPolinomialNorm3 = np.zeros((4, len(C)))
NormFeatures = mu.z_score(features)
dataPCA5Norm = dr.PCA(NormFeatures, 5)
dataPCA5 = dr.PCA(features, 5)

#no Norm no PCA
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF = val.k_fold(Poly3SVMObj, features, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 3) SVM no PCA no Norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[0] = minDCFList.mean(axis=0)

#Norm no PCA
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF = val.k_fold(Poly3SVMObj, NormFeatures, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 3) SVM no PCA Z-norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[1] = minDCFList.mean(axis=0)

#No Norm PCA 5
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF = val.k_fold(Poly3SVMObj, dataPCA5, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 3) SVM PCA 5, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[2] = minDCFList.mean(axis=0)

#Norm PCA 5
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF = val.k_fold(Poly3SVMObj, dataPCA5Norm, labels, 5, (pi, 1, 1))
        print("Polynomial (d = 3) SVM PCA 5 Z-norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[3] = minDCFList.mean(axis=0)

# #Test ZNorm best SVM
# NormFeatures = mu.z_score(features)
# dataPCA5Norm = dr.PCA(NormFeatures, 5)

# for _piT in [0.17, 0.1, 0.2, 0.5]:
#     #RBF 
#     print("######## piT = {} #######".format(_piT))
#     print("------ RBF -------")
#     RBFObj = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=_piT)
#     _, minDCF5 = val.k_fold(RBFObj, dataPCA5, labels, 5, (0.5, 1, 1))
#     _, minDCF1 = val.k_fold(RBFObj, dataPCA5, labels, 5, (0.1, 1, 1))
#     print("minDCF 0.5 RBF: ", minDCF5)
#     print("minDCF 0.1 RBF: ", minDCF1)
#     print("Cprim RBF:", (minDCF5+minDCF1)/2)

#     #Poli2
#     print("------ Poli2 -------")
#     Poly2SVMObj = svm.SVM('Polinomial', balanced=True, d=2, K=10, C=0.01, c=0.1, piT=_piT)
#     _, minDCF5 = val.k_fold(Poly2SVMObj, dataPCA5, labels, 5, (0.5, 1, 1))
#     _, minDCF1 = val.k_fold(Poly2SVMObj, dataPCA5, labels, 5, (0.1, 1, 1))
#     print("minDCF 0.5 Poli2: ", minDCF5)
#     print("minDCF 0.1 Poli2: ", minDCF1)
#     print("Cprim Poli2:", (minDCF5+minDCF1)/2)
