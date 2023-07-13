import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
import sys
from itertools import repeat

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if __name__ == "__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

    #DICTIONARIES
    labels_dict = {
        "Not-Italian": 0,
        "Italian": 1
    }
    features_dict = {
        "feature 0": 0,
        "feature 1": 1,
        "feature 2": 2,
        "feature 3": 3,
        "feature 4": 4,
        "feature 5": 5        
    }
    features_dict_PCA = {
        "PC-0": 0,
        "PC-1": 1,
        "PC-2": 2,
        "PC-3": 3,
        "PC-4": 4   
    }


    dataPCA = dr.PCA(features, 2)
    dv.get_scatter(dataPCA, labels, labels_dict, {"PC-0":0, "PC-1":1})
    # #MVG
    # for pi in [0.1,0.5]:
    #     #NO PCA
    #     mvgObj = gc.multivariate_cl([1-pi, pi])
    #     _, minDCF = val.k_fold(mvgObj, features, labels, 5, (pi, 1, 1))
    #     print("MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
    #     #PCA
    #     for nPCA in [5,4,3]:
    #         dataPCA = dr.PCA(features, nPCA)
    #         mvgObj = gc.multivariate_cl([1-pi, pi])
    #         _, minDCF = val.k_fold(mvgObj, dataPCA, labels, 5, (pi, 1, 1))
    #         print("MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))
    """
    #NAIVE MVG
    for pi in [0.1,0.5]:
        #NO PCA
        naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(naiveMvgObj, features, labels, 5, (pi, 1, 1))
        print("Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(naiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

    #TIED MVG
    for pi in [0.1,0.5]:
        #NO PCA
        tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedMvgObj, features, labels, 5, (pi, 1, 1))
        print("Tied MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(tiedMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Tied MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

    #TIED NAIVE MVG
    for pi in [0.1,0.5]:
        #NO PCA
        tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedNaiveMvgObj, features, labels, 5, (pi, 1, 1))
        print("Tied Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(tiedNaiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Tied Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))
    
    #LOGISTIC REGRESSION
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTrainQuadraticZNorm = mu.z_score(featuresTrainQuadratic)
    featuresZNorm = mu.z_score(features)
    
    #QLOG REG NO NORMALIZATION
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((2, len(lambdas)))
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, False)
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (pi, 1, 1))
            print("LogReg, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)

    #QLOG REG Z-NORM
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, 5, (pi, 1, 1))
            print("LogReg Z-Norm, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)

    dv.plotCPrim(lambdas, CprimLogReg, ["QLog-Reg", "QLog-Reg z-norm"] , "λ", "QLogReg_QLogRegNorm")

    #LOG REG NO NORMALIZATION
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((2, len(lambdas)))
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, False)
            _, minDCF = val.k_fold(logRegObj, features, labels, 5, (pi, 1, 1))
            print("LogReg, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)

    #LOG REG Z-NORM
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, featuresZNorm, labels, 5, (pi, 1, 1))
            print("LogReg Z-Norm, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)

    dv.plotCPrim(lambdas, CprimLogReg, ["Log-Reg", "Log-Reg z-norm"] , "λ", "LogReg_LogRegNorm")
    """
    """
    #LOG REG BALANCED PCA
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((3, len(lambdas)))
    
    for PCAIndex, nPCA in enumerate([5,4]):
        minDCFList = np.zeros((2, len(lambdas)))
        dataPCA = dr.PCA(features, nPCA)
        for index, pi in enumerate([0.1,0.5]):
            for lIndex, l in enumerate(lambdas):
                logRegObj = lrc.logReg(l, pi, True)
                _, minDCF = val.k_fold(logRegObj, dataPCA, labels, 5, (pi, 1, 1))
                print("LogReg Balanced, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
                minDCFList[index, lIndex] = minDCF
        CprimLogReg[PCAIndex] = minDCFList.mean(axis=0)


    #LOG REG NO NORMALIZATION
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, features, labels, 5, (pi, 1, 1))
            print("LogReg Balanced No Normalization, minDCF with pi {} no PCA and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[2] = minDCFList.mean(axis=0)

    print(CprimLogReg)
    dv.plotCPrim(lambdas, CprimLogReg, ["Log-Reg Balanced PCA 5", "Log-Reg Balanced PCA 4", "Log-Reg Balanced no PCA"] , "λ", "LogRegBalancedPCAs")
    """
    """TEST GMM
    dim_target=1
    dim_non_target=32
    GMMclass=gmm.GMM(dim_target,dim_non_target,"diagonal","mvg")
    _,minDCF = val.k_fold(GMMclass,features,labels,5, (0.5,1,1))
    print("GMM {}, {}: {}".format(dim_target,dim_non_target,minDCF))
    """
    
    """print(du.explained_variance(features ))"""
    """C = 10
    CF = 0
    CT = 1
    labels  = [-1,1,1,1,-1,-1,1,1,-1,1]
    bounds = list(repeat((0, C), 10))
    for index,l in enumerate(labels):
        if(labels[index]==1):
            bounds[index] = (0,CT)
        elif(labels[index]==-1):  
            bounds[index]= (0,CF)
    
    print(bounds)"""
    """
    print(f"piEmpT {(features[:,labels == 1].shape[1]/features.shape[1])}")
    print(f"piEmpTF {(features[:,labels == 0].shape[1]/features.shape[1])}")
    """ 
    
    
    # #SVM LINEARE PCA no Norm
    # C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # CprimSVMLin = np.zeros((3, len(C)))
      
    # with open("output_GMM_minDCF.txt", "w") as f:
    
    #     for pi in [0.1,0.5]:
    #         for nTarget in [2,4,8]:
    #             for nNonTarget in [2,4,8,16,32]:
    #                 for MtypeTarget in ["mvg","tied","diagonal","tied diagonal"]:
    #                     for MtypeNonTarget in ["mvg","tied","diagonal","tied diagonal"]:
    #                         GMMClass = gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
    #                         _, minDCF = val.k_fold(GMMClass, features, labels, 5, (pi, 1, 1))
    #                         print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget, pi, minDCF),file=f)
    
    #     for pi in [0.1,0.5]:
    #         for nPCA in [5,4]:
    #                 dataPCA = dr.PCA(features, nPCA)
    #                 for nTarget in [2,4,8]:
    #                     for nNonTarget in [2,4,8,16,32]:
    #                         for MtypeTarget in ["mvg","tied","diagonal","tied diagonal"]:
    #                             for MtypeNonTarget in ["mvg","tied","diagonal","tied diagonal"]:
    #                                 GMMClass = gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
    #                                 _, minDCF = val.k_fold(GMMClass, dataPCA, labels, 5, (pi, 1, 1))
    #                                 print("GMM, minDCF with PCA {}, with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(nPCA,nTarget,nNonTarget,MtypeTarget,MtypeNonTarget, pi, minDCF),file=f)

    # GMMClass = gmm.GMM(1,32,'diagonal','diagonal')
    # _, minDCF = val.k_fold(GMMClass, features, labels, 5, (0.1, 1, 1))
    # print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(1,32,'diagonal','tied', 0.1, minDCF))

    # GMMClass = gmm.GMM(1,32,'diagonal','diagonal')
    # _, minDCF = val.k_fold(GMMClass, features, labels, 5, (0.5, 1, 1))
    # print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(1,32,'diagonal','tied', 0.5, minDCF))

    # GMMClass = gmm.GMM(1,32,'diagonal','tied')
    # _, minDCF = val.k_fold(GMMClass, features, labels, 5, (0.1, 1, 1))
    # print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(1,32,'diagonal','tied diagonal', 0.1, minDCF))

    # GMMClass = gmm.GMM(1,32,'diagonal','tied')
    # _, minDCF = val.k_fold(GMMClass, features, labels, 5, (0.5, 1, 1))
    # print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(1,32,'diagonal','tied diagonal', 0.5, minDCF))

    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")
        