import numpy as np
import data_utils as du
from sklearn.utils import shuffle
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import SVM_classifiers as svm
import GMM as gmm
import validation as val

if __name__ == "__main__":

    L, D = du.load(".\Data\Train.txt")
    LT, DT = du.load(".\Data\Test.txt")

    # # CHOSEN
    svmc = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    qlr = lrc.logReg(10,0.1,"balanced")
    gmmc = gmm.GMM(2,32,"MVG", "tied")
    train_set_svm = dr.PCA(D, 5)
    train_set_qlr = du.features_expansion(D)
    train_set_gmm = D
    test_set_svm = dr.PCA(DT, 5)
    test_set_qlr = du.features_expansion(DT)
    test_set_gmm = DT

    # OPTIMAL 
    # svmc = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=1, piT=0.1)
    # qlr = lrc.logReg(1,0.17,"balanced")
    # gmmc = gmm.GMM(2,32,"diagonal", "tied diagonal")
    # train_set_svm = D
    # train_set_qlr = du.features_expansion(D)
    # train_set_gmm = D
    # test_set_svm = DT
    # test_set_qlr = du.features_expansion(DT)
    # test_set_gmm = DT


    # BEST FUSION
    FUSER = lrc.logReg(0.001, 0.4, "balanced")

    qlr.train(train_set_qlr, L)
    svmc.train(train_set_svm, L)
    gmmc.train(train_set_gmm, L)

    predictedQLR = qlr.transform(test_set_qlr)
    predicetedSVM = svmc.transform(test_set_svm)
    predictedGMM = gmmc.transform(test_set_gmm)

    scoresQLR = qlr.get_scores()
    scoresSVM = svmc.get_scores()
    scoresGMM = gmmc.get_scores()

    scoresQGS = np.vstack((scoresQLR, scoresGMM, scoresSVM))
    scoresQG = np.vstack((scoresQLR, scoresGMM))
    scoresQS = np.vstack((scoresQLR, scoresSVM))
    scoresGS = np.vstack((scoresGMM, scoresSVM))

    actualDCF1FusionQGS, minDCF1FusionQGS, _, _ = val.k_fold_bayes_plot(FUSER, scoresQGS, LT, 5, (0.1, 1, 1), "QGS", False)
    actualDCF2FusionQGS, minDCF2FusionQGS, scoresQGS, predictedQGS = val.k_fold_bayes_plot(FUSER, scoresQGS, LT, 5, (0.5, 1, 1), "QGS", False)
    actualDCF1FusionQG, minDCF1FusionQG, _, _ = val.k_fold_bayes_plot(FUSER, scoresQG, LT, 5, (0.1, 1, 1), "QG", False)
    actualDCF2FusionQG, minDCF2FusionQG, _, _ = val.k_fold_bayes_plot(FUSER, scoresQG, LT, 5, (0.5, 1, 1), "QG", False)
    actualDCF1FusionQS, minDCF1FusionQS, _, _ = val.k_fold_bayes_plot(FUSER, scoresQS, LT, 5, (0.1, 1, 1), "QS", False)
    actualDCF2FusionQS, minDCF2FusionQS, _, _ = val.k_fold_bayes_plot(FUSER, scoresQS, LT, 5, (0.5, 1, 1), "QS", False)
    actualDCF1FusionGS, minDCF1FusionGS, _, _ = val.k_fold_bayes_plot(FUSER, scoresGS, LT, 5, (0.1, 1, 1), "GS", False)
    actualDCF2FusionGS, minDCF2FusionGS, _, _ = val.k_fold_bayes_plot(FUSER, scoresGS, LT, 5, (0.5, 1, 1), "GS", False)     

    minCprimFusionQGS = (minDCF1FusionQGS+minDCF2FusionQGS)/2
    CprimFusionQGS = (actualDCF1FusionQGS+actualDCF2FusionQGS)/2
    minCprimFusionQG = (minDCF1FusionQG+minDCF2FusionQG)/2
    CprimFusionQG = (actualDCF1FusionQG+actualDCF2FusionQG)/2
    minCprimFusionQS = (minDCF1FusionQS+minDCF2FusionQS)/2
    CprimFusionQS = (actualDCF1FusionQS+actualDCF2FusionQS)/2
    minCprimFusionGS = (minDCF1FusionGS+minDCF2FusionGS)/2
    CprimFusionGS = (actualDCF1FusionGS+actualDCF2FusionGS)/2

    #Printing results
    print("minDCF Fusion QGS 0.1:", minDCF1FusionQGS)
    print("minDCF Fusion QGS 0.5:", minDCF2FusionQGS)
    print("actDCF Fusion QGS 0.1:", actualDCF1FusionQGS)
    print("actDCF Fusion QGS 0.5:", actualDCF2FusionQGS)
    print("minCprim Fusion QGS:", minCprimFusionQGS)
    print("Cprim Fusion QGS:", CprimFusionQGS)

    print("minDCF Fusion QG 0.1:", minDCF1FusionQG)
    print("minDCF Fusion QG 0.5:", minDCF2FusionQG)
    print("actDCF Fusion QG 0.1:", actualDCF1FusionQG)
    print("actDCF Fusion QG 0.5:", actualDCF2FusionQG)
    print("minCprim Fusion QG:", minCprimFusionQG)
    print("Cprim Fusion QG:", CprimFusionQG)

    print("minDCF Fusion QS 0.1:", minDCF1FusionQS)
    print("minDCF Fusion QS 0.5:", minDCF2FusionQS)
    print("actDCF Fusion QS 0.1:", actualDCF1FusionQS)
    print("actDCF Fusion QS 0.5:", actualDCF2FusionQS)
    print("minCprim Fusion QS:", minCprimFusionQS)
    print("Cprim Fusion QS:", CprimFusionQS)

    print("minDCF Fusion GS 0.1:", minDCF1FusionGS)
    print("minDCF Fusion GS 0.5:", minDCF2FusionGS)
    print("actDCF Fusion GS 0.1:", actualDCF1FusionGS)
    print("actDCF Fusion GS 0.5:", actualDCF2FusionGS)
    print("minCprim Fusion GS:", minCprimFusionGS)
    print("Cprim Fusion GS:", CprimFusionGS)

    # PLOTTING
    # CalibratedScoresGMM = np.load("CalibratedScoresGMM.npy")
    # CalibratedScoresQLR = np.load("CalibratedScoresQLR.npy")
    # CalibratedScoresSVM = np.load("CalibratedScoresSVM.npy")

    # predictedSVM = np.load("PredictedSVM.npy")
    # predictedQLR = np.load("PredictedQLR.npy")
    # predictedGMM = np.load("PredictedGMM.npy")

    # legend = ["Fusion actDCF", "Fusion minDCF", "SVM actDCF", "SVM minDCF", "QLR actDCF", "QLR minDCF", "GMM actDCF", "GMM minDCF"]
    # val.get_multi_error_plot_fusion([scoresQGS, CalibratedScoresSVM, CalibratedScoresQLR, CalibratedScoresGMM], 1,1,LT, [predictedQGS, predictedSVM, predictedQLR, predictedGMM], legend, "Calibrated_eval_chosen")
    