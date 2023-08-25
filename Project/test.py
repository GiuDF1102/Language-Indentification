import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation2 as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
import sys
from itertools import repeat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import fusion

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if __name__ == "__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load("Train.txt")
    labels_test, features_test = du.load("Test.txt")
    labels_sh = shuffle(labels,random_state=0)
    features_expanded = du.features_expansion(features)
    features_PCA_5 = dr.PCA(features, 5)

    #BEST MODELS
    QLR = lrc.logReg(10, 0.1, "balanced")
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    GMM = gmm.GMM(2, 32, "mvg", "tied")

    """
    ### DET PLOTS
    # BEST MODELS
    ## Quadratic Logistic Regression NO PCA piT = 0.1, lambda = 10 
    _,_,scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.5, 1, 1), "QLR")

    ## SVM RBF Balanced gamma = 0.01, C = 0.1, K = 0.01 piT = 0.2 PCA 5
    _,_,scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.5, 1, 1), "SVM")
    
    ## GMM 2 FC 32 FC - T NO PCA
    _,_,scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM")

    val.get_multi_DET([scoresQLR, scoresSVM, scoresGMM], labels_sh, ["QLR", "SMV", "GMM"], "Best Models")
    """

    ### FUSION
    FUSER = lrc.logReg(0, 0.1, "balanced")
    pi = 0.1

    # TRAIN MODELS AND GET SCORES
    _,_,scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (pi, 1, 1), "QLR")
    _,_,scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (pi, 1, 1), "SVM")
    _,_,scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (pi, 1, 1), "GMM")

    # STACK THE SCORES
    scoresQSG = np.vstack((scoresQLR, scoresSVM, scoresGMM))
    scoresQS = np.vstack((scoresQLR, scoresSVM))
    scoresQG = np.vstack((scoresQLR, scoresGMM))
    scoresSG = np.vstack((scoresSVM, scoresGMM))

    # SEND THE SCORES TO THE LOG REG TO OBAIN THE WEIGHTS
    FUSER.train(scoresQSG, labels_sh)
    wQSG, bQSG = FUSER.get_params()
    FUSER.train(scoresQS, labels_sh)
    wQS, bQS = FUSER.get_params()
    FUSER.train(scoresQG, labels_sh)
    wQG, bQG = FUSER.get_params()
    FUSER.train(scoresSG, labels_sh)
    wSG, bSG = FUSER.get_params()


    # USE WEIGHTS FOR CLASSIFICATION
    predictedQSG = np.where(wQSG.T.dot(scoresQSG) + bQSG > np.log(pi/(1-pi)), 1, 0)
    predictedQS = np.where(wQS.T.dot(scoresQS) + bQS > np.log(pi/(1-pi)), 1, 0)
    predictedQG = np.where(wQG.T.dot(scoresQG) + bQG > np.log(pi/(1-pi)), 1, 0)
    predictedSG = np.where(wSG.T.dot(scoresSG) + bSG > np.log(pi/(1-pi)), 1, 0)

    # GET ADCF AND MINDCF
    minDCFQSG = val.min_DCF(scoresQSG[0], pi, 1, 1, labels_sh, predictedQSG)
    actualDCFQSG = val.act_DCF(scoresQSG[0], pi, 1, 1, labels_sh, None)
    minDCFQS = val.min_DCF(scoresQS[0], pi, 1, 1, labels_sh, predictedQS)
    actualDCFQS = val.act_DCF(scoresQS[0], pi, 1, 1, labels_sh, None)
    minDCFQG = val.min_DCF(scoresQG[0], pi, 1, 1, labels_sh, predictedQG)
    actualDCFQG = val.act_DCF(scoresQG[0], pi, 1, 1, labels_sh, None)
    minDCFSG = val.min_DCF(scoresSG[0], pi, 1, 1, labels_sh, predictedSG)
    actualDCFSG = val.act_DCF(scoresSG[0], pi, 1, 1, labels_sh, None)

    # PRINT RESULTS
    print(f"QSG minDCF: {minDCFQSG} actualDCF: {actualDCFQSG}")
    print(f"QS minDCF: {minDCFQS} actualDCF: {actualDCFQS}")
    print(f"QG minDCF: {minDCFQG} actualDCF: {actualDCFQG}")
    print(f"SG minDCF: {minDCFSG} actualDCF: {actualDCFSG}")

    """
    Fusion results
        cal_l = 0 pi = 0.5
            QSG minDCF: 0.10258878741755453 actualDCF: 0.20333840690005073
            QS minDCF: 0.10258878741755453 actualDCF: 0.20333840690005073
            QG minDCF: 0.10258878741755453 actualDCF: 0.20333840690005073
            SG minDCF: 0.0889269406392694 actualDCF: 0.13547818366311515

        cal_l = 0 pi = 0.1
            QSG minDCF: 0.3809132420091324 actualDCF: 1.0
            QS minDCF: 0.3809132420091324 actualDCF: 1.0
            QG minDCF: 0.3809132420091324 actualDCF: 1.0
            SG minDCF: 0.3717808219178082 actualDCF: 1.0

        
        QSG minCprim: 0.241 Cprim: 0.601
        QS minCprim: 0.241 Cprim: 0.601
        QG minCprim: 0.241 Cprim: 0.601
        SG minCprim: 0.230 Cprim: 0.567
    """

    end_time = datetime.now()
    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")

