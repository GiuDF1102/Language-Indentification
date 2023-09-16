import validation as val
import data_utils as du
import dimensionality_reduction as dr
import SVM_classifiers as svm
import math_utils as mu
import numpy as np
import data_visualization as dv

#LOADING DATASET
labels, features = du.load(".\Data\Train.txt")
labels_test, features_test = du.load(".\Data\Test.txt")
featuresZNorm = mu.z_score(features)
C = np.logspace(-3, 5, num=9)

# #LINEAR SVM
# CprimLinearNorm = np.zeros((2, len(C)))
# ##LINEAR SVM NOT NORMALIZED NOT REBALANCED
# minDCFList = np.zeros((2, len(C)))
# for index, pi in enumerate([0.1,0.5]):
#     for cIndex, c in enumerate(C):
#         SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#         _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, features, labels, 5, (pi, 1, 1), "Linear SVM", False)
#         print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#         minDCFList[index, cIndex] = minDCF
# CprimLinearNorm[0] = minDCFList.mean(axis=0)

# ##LINEAR SVM NORMALIZED NOT REBALANCED
# minDCFList = np.zeros((2, len(C)))
# for index, pi in enumerate([0.1,0.5]):
#     for cIndex, c in enumerate(C):
#         SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#         _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, featuresZNorm, labels, 5, (pi, 1, 1), "Linear SVM Z-norm", False)
#         print("Linear SVM Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#         minDCFList[index, cIndex] = minDCF
# CprimLinearNorm[1] = minDCFList.mean(axis=0)

# dv.plotCPrim(C, CprimLinearNorm, ["Linear SVM", "Linear SVM Z-norm"] , "C", "LinearSVM_LinearSVMNormGIUSTO")

#LINEAR SVM PCA
#CprimLinearNorm = np.zeros((3, len(C)))
##LINEAR SVM PCA NOT NORMALIZED NOT REBALANCED
# for PCAIndex, nPCA in enumerate([5,4]):
#     minDCFList = np.zeros((2, len(C)))
#     dataPCA = dr.PCA(features, nPCA)
#     for index, pi in enumerate([0.1,0.5]):
#         for cIndex, c in enumerate(C):
#             SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#             _, minDCF = val.k_fold(SVMObj, dataPCA, labels, 5, (pi, 1, 1))
#             print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#             minDCFList[index, cIndex] = minDCF
#     CprimLinearNorm[PCAIndex] = minDCFList.mean(axis=0)

# ##LINEAR SVM NOT NORMALIZED NOT REBALANCED
# minDCFList = np.zeros((2, len(C)))
# for index, pi in enumerate([0.1,0.5]):
#     for cIndex, c in enumerate(C):
#         SVMObj = svm.SVM('linear', balanced=False, K=1, C=c)
#         _, minDCF = val.k_fold(SVMObj, features, labels, 5, (pi, 1, 1))
#         print("Linear SVM Not Normalized, minDCF with pi {} and C {} is {}".format(pi, c, minDCF))
#         minDCFList[index, cIndex] = minDCF
# CprimLinearNorm[2] = minDCFList.mean(axis=0)

# dv.plotCPrim(C, CprimLinearNorm, ["Linear SVM PCA 5", "Linear SVM PCA 4", "Linear SVM no PCA"] , "C", "LinearSVMPCAs")


##################################################

## POLYNOMIAL 2 SVM
namesList = [
    "Polinomial(2) SVM c = 0.1",
    "Polinomial(2) SVM c = 1",
    "Polinomial(2) SVM c = 10"    
]

dataPCA5 = dr.PCA(features, 5)
dataPCA4 = dr.PCA(features, 4)
k = 5

with open("output_SVM_Polinomial2_minDCFAndCprim.txt", "w") as f:

    for K_ in [1,10,100]:
        Cprimlists = []
        for c_ in [0.1,1,10]:
            CprimlistForGamma = []
            for C_ in [0.001,0.01,0.1,1,10,100]:
                minDCFList=[]
                for pi in [0.1,0.5]:
                    SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=K_, C=C_, d=2)
                    _, minDCF = val.k_fold(SVMObj, features, labels, k, (pi, 1, 1))
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF),file=f)
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF))
                    minDCFList.append(minDCF)
                Cprim=np.array(minDCFList).mean(axis=0)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim),file=f)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim))
                CprimlistForGamma.append(Cprim)
                print(CprimlistForGamma)
            if c_ in [0.1,1,10]:
                Cprimlists.append(CprimlistForGamma)
        dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"Polinomial2NoPCA K = {K_}")

    for K_ in [1,10,100]:
        Cprimlists = []
        for c_ in [0.1,1,10]:
            CprimlistForGamma = []
            for C_ in [0.001,0.01,0.1,1,10,100]:
                minDCFList=[]
                for pi in [0.1,0.5]:
                    SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=K_, C=C_, d=2)
                    _, minDCF = val.k_fold(SVMObj, dataPCA5, labels, k, (pi, 1, 1))
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF),file=f)
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,K_, minDCF))
                    minDCFList.append(minDCF)
                Cprim=np.array(minDCFList).mean(axis=0)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim),file=f)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and c:{} is {}".format(C_,K_,c_,Cprim))
                CprimlistForGamma.append(Cprim)
                print(CprimlistForGamma)
            if c_ in [0.1,1,10]:
                Cprimlists.append(CprimlistForGamma)
        dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"Polinomial2PCA5 K = {K_}")

## POLYNOMIAL 3 SVM
C = [0.00001,0.0001,0.001,0.01, 0.1, 1, 10]
CprimPolinomialNorm3 = np.zeros((4, len(C)))
NormFeatures = mu.z_score(features)
dataPCA5Norm = dr.PCA(NormFeatures, 5)
dataPCA5 = dr.PCA(features, 5)

#no Norm no PCA
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF, _, _ = val.k_fold_bayes_plot(Poly3SVMObj, features, labels, 5, (pi, 1, 1), "Polynomial (d = 3) SVM", False)
        print("Polynomial (d = 3) SVM no PCA no Norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[0] = minDCFList.mean(axis=0)

#Norm no PCA
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF, _, _ = val.k_fold_bayes_plot(Poly3SVMObj, NormFeatures, labels, 5, (pi, 1, 1), "Polynomial (d = 3) SVM Z-norm", False)
        print("Polynomial (d = 3) SVM no PCA Z-norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[1] = minDCFList.mean(axis=0)

#No Norm PCA 5
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF, _, _ = val.k_fold_bayes_plot(Poly3SVMObj, dataPCA5, labels, 5, (pi, 1, 1), "Polynomial (d = 3) SVM PCA 5", False)
        print("Polynomial (d = 3) SVM PCA 5, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[2] = minDCFList.mean(axis=0)

#Norm PCA 5
minDCFList = np.zeros((2, len(C)))
for index, pi in enumerate([0.1,0.5]):
    for cIndex, C_ in enumerate(C):
        Poly3SVMObj = svm.SVM('Polinomial', balanced=False, d=3, K=10, C=C_, c=0.1)
        _, minDCF, _, _ = val.k_fold_bayes_plot(Poly3SVMObj, dataPCA5Norm, labels, 5, (pi, 1, 1), "Polynomial (d = 3) SVM PCA 5 Z-norm", False)
        print("Polynomial (d = 3) SVM PCA 5 Z-norm, minDCF with pi {} and C {} is {}".format(pi, C_, minDCF))
        minDCFList[index, cIndex] = minDCF
CprimPolinomialNorm3[3] = minDCFList.mean(axis=0)

dv.plotCPrim(C, CprimPolinomialNorm3, ["Polynomial (d = 3) SVM", "Polynomial (d = 3) SVM Z-Norm", "Polynomial (d = 3) SVM PCA 5", "Polynomial (d = 3) SVM PCA 5 Z-Norm"] , "C", "Polynomial3SVMPCAsNormGIUSTO")

## POLINOMIAL2 SVM NORM PCA BEST K
with open("output_SVM_Poli_minDCFAndCprimNORMPCA5.txt", "w") as f:
    NormFeatures = mu.z_score(features)
    dataPCA5Norm = dr.PCA(NormFeatures, 5)

    namesList = [
        "Polinomial(2) SVM c = 0.1",
        "Polinomial(2) SVM c = 1",
        "Polinomial(2) SVM c = 10",
    ]

    Cprimlists = []
    for c_ in [0.1, 1, 10]:
        CprimlistForc = []
        for C_ in [0.001, 0.01,0.1, 1, 10, 100]:
            minDCFList=[]
            for pi in [0.1,0.5]:
                SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=10, C=C_, d=2)
                _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, dataPCA5Norm, labels, k, (pi, 1, 1), "RBF SVM", False)
                print("Poly SVM Not Balanced Normalized PCA 5, minDCF and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,10, minDCF),file=f)
                print("Poly SVM Not Balanced Normalized PCA 5, minDCF and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,10, minDCF))
                minDCFList.append(minDCF)
            Cprim=np.array(minDCFList).mean(axis=0)
            print("Poly SVM Not Balanced Normalized PCA 5, Cprim and C: {} and K: {} and c:{} is {}".format(C_,10,c_,Cprim),file=f)
            print("Poly SVM Not Balanced Normalized PCA 5, Cprim and C: {} and K: {} and c:{} is {}".format(C_,10,c_,Cprim))
            CprimlistForc.append(Cprim)
            print(CprimlistForc)
        if c_ in [0.1, 1, 10]:
            Cprimlists.append(CprimlistForc)
    dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"SVMPOLYPCA5NORM K = {10}")
    
## POLINOMIAL2 SVM NORM BEST K
with open("output_SVM_Poli_minDCFAndCprimNORM.txt", "w") as f:
    NormFeatures = mu.z_score(features)
    namesList = [
        "Polinomial(2) SVM c = 0.1",
        "Polinomial(2) SVM c = 1",
        "Polinomial(2) SVM c = 10",
    ]

    Cprimlists = []
    for c_ in [0.1, 1, 10]:
        CprimlistForc = []
        for C_ in [0.001, 0.01,0.1, 1, 10, 100]:
            minDCFList=[]
            for pi in [0.1,0.5]:
                SVMObj = svm.SVM('Polinomial', balanced=False,c=c_, K=10, C=C_, d=2)
                _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, NormFeatures, labels, k, (pi, 1, 1), "RBF SVM", False)
                print("Poly SVM Not Balanced Normalized, minDCF and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,10, minDCF),file=f)
                print("Poly SVM Not Balanced Normalized, minDCF and pi: {} and C: {} and c:{} and K: {} is {}".format(pi, C_,c_,10, minDCF))
                minDCFList.append(minDCF)
            Cprim=np.array(minDCFList).mean(axis=0)
            print("Poly SVM Not Balanced Normalized, Cprim and C: {} and K: {} and c:{} is {}".format(C_,10,c_,Cprim),file=f)
            print("Poly SVM Not Balanced Normalized, Cprim and C: {} and K: {} and c:{} is {}".format(C_,10,c_,Cprim))
            CprimlistForc.append(Cprim)
            print(CprimlistForc)
        if c_ in [0.1, 1, 10]:
            Cprimlists.append(CprimlistForc)
    dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"SVMPOLYNORM K = {10}")

#####################################################

## RBF SVM
namesList = [
    "RBF SVM γ = 0.001",
    "RBF SVM γ = 0.01",
    "RBF SVM γ = 0.1"    
]

dataPCA5 = dr.PCA(features, 5)
dataPCA4 = dr.PCA(features, 4)
k = 5

with open("output_SVM_RBF_minDCFAndCprim.txt", "w") as f:

    for K_ in [1,10,100]:
        Cprimlists = []
        for gamma_ in [0.001,0.01,0.1]:
            CprimlistForGamma = []
            for C_ in [0.001,0.01,0.1,1,10,100]:
                minDCFList=[]
                for pi in [0.1,0.5]:
                    SVMObj = svm.SVM('RBF', balanced=False,gamma=gamma_, K=K_, C=C_)
                    _, minDCF = val.k_fold(SVMObj, features, labels, k, (pi, 1, 1))
                    print("RBF SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,K_, minDCF),file=f)
                    print("RBF SVM Not Balanced Not Normalized, minDCF NO PCA and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,K_, minDCF))
                    minDCFList.append(minDCF)
                Cprim=np.array(minDCFList).mean(axis=0)
                print("RBF SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and gamma:{} is {}".format(C_,K_,gamma_,Cprim),file=f)
                print("RBF SVM Not Balanced Not Normalized, Cprim NO PCA and C: {} and K: {} and gamma:{} is {}".format(C_,K_,gamma_,Cprim))
                CprimlistForGamma.append(Cprim)
                print(CprimlistForGamma)
            if gamma_ in [0.001,0.01,0.1]:
                Cprimlists.append(CprimlistForGamma)
        dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"RBFNoPCA K = {K_}")

    for K_ in [1,10,100]:
        Cprimlists = []
        for gamma_ in [0.001,0.01,0.1]:
            CprimlistForGamma = []
            for C_ in [0.001,0.01,0.1,1,10,100]:
                minDCFList=[]
                for pi in [0.1,0.5]:
                    SVMObj = svm.SVM('RBF', balanced=False,gamma=gamma_, K=K_, C=C_)
                    _, minDCF = val.k_fold(SVMObj, dataPCA5, labels, k, (pi, 1, 1))
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,K_, minDCF),file=f)
                    print("Polinomial SVM Not Balanced Not Normalized, minDCF PCA 5 and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,K_, minDCF))
                    minDCFList.append(minDCF)
                Cprim=np.array(minDCFList).mean(axis=0)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and gamma:{} is {}".format(C_,K_,gamma_,Cprim),file=f)
                print("Polinomial SVM Not Balanced Not Normalized, Cprim PCA 5 and C: {} and K: {} and gamma:{} is {}".format(C_,K_,gamma_,Cprim))
                CprimlistForGamma.append(Cprim)
                print(CprimlistForGamma)
            if gamma_ in [0.001,0.01,0.1]:
                Cprimlists.append(CprimlistForGamma)
        dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"RBFPCA5 K = {K_}")

# NORM BEST K
with open("output_SVM_RBF_minDCFAndCprimNORM2.txt", "w") as f:
    NormFeatures = mu.z_score(features)
    namesList = [
        "RBF SVM gamma = 0.001",
        "RBF SVM gamma = 0.01",
        "RBF SVM gamma = 0.1"
    ]
    Cprimlists = []
    for gamma_ in [0.001, 0.01, 0.1]:
        CprimlistForGamma = []
        for C_ in [100, 1000,10000]:
            minDCFList=[]
            for pi in [0.1,0.5]:
                SVMObj = svm.SVM('RBF', balanced=False,gamma=gamma_, K=0.01, C=C_)
                _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, NormFeatures, labels, k, (pi, 1, 1), "RBF SVM", False)
                print("RBF SVM Not Balanced Normalized, minDCF and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,0.01, minDCF),file=f)
                print("RBF SVM Not Balanced Normalized, minDCF and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,0.01, minDCF))
                minDCFList.append(minDCF)
            Cprim=np.array(minDCFList).mean(axis=0)
            print("RBF SVM Not Balanced Normalized, Cprim and C: {} and K: {} and gamma:{} is {}".format(C_,0.01,gamma_,Cprim),file=f)
            print("RBF SVM Not Balanced Normalized, Cprim and C: {} and K: {} and gamma:{} is {}".format(C_,0.01,gamma_,Cprim))
            CprimlistForGamma.append(Cprim)
            print(CprimlistForGamma)
        if gamma_ in [0.001,0.01,0.1]:
            Cprimlists.append(CprimlistForGamma)
    dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"SVMRBFNORM2 K = {0.01}")

# NORM PCA BEST K
with open("output_SVM_RBF_minDCFAndCprimNORMPCA5.txt", "w") as f:
    NormFeatures = mu.z_score(features)
    dataPCA5Norm = dr.PCA(NormFeatures, 5)
    namesList = [
        "RBF SVM gamma = 0.001",
        "RBF SVM gamma = 0.01",
        "RBF SVM gamma = 0.1"
    ]
    Cprimlists = []
    for gamma_ in [0.001, 0.01, 0.1]:
        CprimlistForGamma = []
        for C_ in [0.001, 0.01,0.1, 1, 10, 100]:
            minDCFList=[]
            for pi in [0.1,0.5]:
                SVMObj = svm.SVM('RBF', balanced=False,gamma=gamma_, K=0.01, C=C_)
                _, minDCF, _, _ = val.k_fold_bayes_plot(SVMObj, dataPCA5Norm, labels, k, (pi, 1, 1), "RBF SVM", False)
                print("RBF SVM Not Balanced Normalized, minDCF PCA 5 and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,0.01, minDCF),file=f)
                print("RBF SVM Not Balanced Normalized, minDCF PCA 5 and pi: {} and C: {} and gamma:{} and K: {} is {}".format(pi, C_,gamma_,0.01, minDCF))
                minDCFList.append(minDCF)
            Cprim=np.array(minDCFList).mean(axis=0)
            print("RBF SVM Not Balanced Normalized, Cprim PCA 5 and C: {} and K: {} and gamma:{} is {}".format(C_,0.01,gamma_,Cprim),file=f)
            print("RBF SVM Not Balanced Normalized, Cprim PCA 5 and C: {} and K: {} and gamma:{} is {}".format(C_,0.01,gamma_,Cprim))
            CprimlistForGamma.append(Cprim)
            print(CprimlistForGamma)
        if gamma_ in [0.001,0.01,0.1]:
            Cprimlists.append(CprimlistForGamma)
    dv.plotCPrim([0.001,0.01,0.1,1,10,100], Cprimlists, namesList, "C", f"SVMRBAFPCA5NORM K = {0.01}")

##################################################### BALANCING

for _piT in [0.17, 0.1, 0.2, 0.5]:
    #RBF 
    print("######## piT = {} #######".format(_piT))
    print("------ RBF -------")
    RBFObj = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=_piT)
    _, minDCF5 = val.k_fold(RBFObj, dataPCA5, labels, 5, (0.5, 1, 1))
    _, minDCF1 = val.k_fold(RBFObj, dataPCA5, labels, 5, (0.1, 1, 1))
    print("minDCF 0.5 RBF: ", minDCF5)
    print("minDCF 0.1 RBF: ", minDCF1)
    print("Cprim RBF:", (minDCF5+minDCF1)/2)

    #Poli2
    print("------ Poli2 -------")
    Poly2SVMObj = svm.SVM('Polinomial', balanced=True, d=2, K=10, C=0.01, c=0.1, piT=_piT)
    _, minDCF5 = val.k_fold(Poly2SVMObj, dataPCA5, labels, 5, (0.5, 1, 1))
    _, minDCF1 = val.k_fold(Poly2SVMObj, dataPCA5, labels, 5, (0.1, 1, 1))
    print("minDCF 0.5 Poli2: ", minDCF5)
    print("minDCF 0.1 Poli2: ", minDCF1)
    print("Cprim Poli2:", (minDCF5+minDCF1)/2)
