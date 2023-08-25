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
    pi = 0.5

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
    scoresFUSQSG = wQSG.T.dot(scoresQSG) + bQSG
    scoresFUSQS = wQS.T.dot(scoresQS) + bQS
    scoresFUSQG = wQG.T.dot(scoresQG) + bQG
    scoresFUSSG = wSG.T.dot(scoresSG) + bSG

    predictedQSG = np.where(scoresFUSQSG > np.log(pi/(1-pi)), 1, 0)
    predictedQS = np.where(scoresFUSQS > np.log(pi/(1-pi)), 1, 0)
    predictedQG = np.where(scoresFUSQG > np.log(pi/(1-pi)), 1, 0)
    predictedSG = np.where(scoresFUSSG > np.log(pi/(1-pi)), 1, 0)

    # GET ADCF AND MINDCF
    minDCFQSG = val.min_DCF(scoresFUSQSG, pi, 1, 1, labels_sh, predictedQSG)
    actualDCFQSG = val.act_DCF(scoresFUSQSG, pi, 1, 1, labels_sh, None)
    minDCFQS = val.min_DCF(scoresFUSQS, pi, 1, 1, labels_sh, predictedQS)
    actualDCFQS = val.act_DCF(scoresFUSQS, pi, 1, 1, labels_sh, None)
    minDCFQG = val.min_DCF(scoresFUSQG, pi, 1, 1, labels_sh, predictedQG)
    actualDCFQG = val.act_DCF(scoresFUSQG, pi, 1, 1, labels_sh, None)
    minDCFSG = val.min_DCF(scoresFUSSG, pi, 1, 1, labels_sh, predictedSG)
    actualDCFSG = val.act_DCF(scoresFUSSG, pi, 1, 1, labels_sh, None)

    # PRINT RESULTS
    print(f"QSG minDCF: {minDCFQSG} actualDCF: {actualDCFQSG}")
    print(f"QS minDCF: {minDCFQS} actualDCF: {actualDCFQS}")
    print(f"QG minDCF: {minDCFQG} actualDCF: {actualDCFQG}")
    print(f"SG minDCF: {minDCFSG} actualDCF: {actualDCFSG}")

    """
    Fusion results
        cal_l = 0 pi = 0.5
        QSG minDCF: 0.085 actualDCF: 0.198
        QS minDCF: 0.091 actualDCF: 0.219
        QG minDCF: 0.085 actualDCF: 0.198
        SG minDCF: 0.084 actualDCF: 0.199

        cal_l = 0 pi = 0.1
            QSG minDCF: 0.348 actualDCF: 0.583
            QS minDCF: 0.370 actualDCF: 0.655
            QG minDCF: 0.346 actualDCF: 0.588
            SG minDCF: 0.346 actualDCF: 0.561

        
        QSG minCprim: 0.216 Cprim: 0.390
        QS minCprim: 0.230 Cprim: 0.442
        QG minCprim: 0.216 Cprim: 0.400
        SG minCprim: 0.215 Cprim: 0.380

        BEST MODEL SVM + GMM
    """

    #val.get_error_plot(scoresFUSSG, 1, 1, labels_sh, predictedSG, "Best fusion")


    end_time = datetime.now()
    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")

