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
    DPCA5 = dr.PCA(D, 5)
    DPCA5T = dr.PCA(DT, 5)
    expanded = du.features_expansion(D)
    expandedT = du.features_expansion(DT)

    # # CHOSEN
    # svmc = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    # qlr = lrc.logReg(10,0.1,"balanced")
    # gmmc = gmm.GMM(2,32,"MVG", "tied")

    # OPTIMAL 
    svmc = svm.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=1, piT=0.1)
    qlr = lrc.logReg(1,0.17,"balanced")
    gmmc = gmm.GMM(2,32,"diagonal", "tied diagonal")

    #GETTING CALIBRATION SET
    print("Getting calibration set SVM...")
    actualDCFSVM1, minDCFSVM1, scoresSVM1, predictedSVM1 = val.k_fold_bayes_plot(svmc, D, L, 5, (0.1, 1, 1), "PROVASVM", False)
    actualDCFSVM2, minDCFSVM2, scoresSVM2, predictedSVM2 = val.k_fold_bayes_plot(svmc, D, L, 5, (0.5, 1, 1), "PROVASVM", False)

    print("Getting calibration set QLR...")
    actualDCFQLR1, minDCFQLR1, scoresQLR1,predictedQLOGREG1 = val.k_fold_bayes_plot(qlr, expanded, L, 5, (0.1, 1, 1),"DETQLOGREG", False)
    actualDCFQLR2, minDCFQLR2, scoresQLR2,predictedQLOGREG2 = val.k_fold_bayes_plot(qlr, expanded, L, 5, (0.5, 1, 1),"DETQLOGREG", False)

    print("Getting calibration set GMM...")
    actualDCFGMM1, minDCFGMM1, scoresGMM1,predictedGMM1 = val.k_fold_bayes_plot(gmmc, D, L, 5, (0.1, 1, 1), "PROVASVM", False)
    actualDCFGMM2, minDCFGMM2, scoresGMM2,predictedGMM2 = val.k_fold_bayes_plot(gmmc, D, L, 5, (0.5, 1, 1), "PROVASVM", False)

    #Scores are the same, calibration sets for each classifier
    calSetSVM = scoresSVM1
    calSetQLR = scoresQLR1
    calSetGMM = scoresGMM1
    L_SH = shuffle(L,random_state=0)


    #Using calibration set to train the calibrators
    print("Training the calibrators...")
    calSetSVM = np.reshape(calSetSVM,(1,len(calSetSVM)))
    calSetQLR = np.reshape(calSetQLR,(1,len(calSetQLR)))
    calSetGMM = np.reshape(calSetGMM,(1,len(calSetGMM)))
    #SVM
    calibratorSVM = lrc.logReg(0,0.5,"balanced")
    calibratorSVM.train(calSetSVM,L_SH)
    #QLR
    calibratorQLR = lrc.logReg(0,0.5,"balanced")
    calibratorQLR.train(calSetQLR,L_SH)
    #GMM
    calibratorGMM = lrc.logReg(0,0.5,"balanced")
    calibratorGMM.train(calSetGMM,L_SH)

    #Training the classifiers with the whole training set
    print("Training the classifiers with the whole training set...")
    svmc.train(D,L)
    qlr.train(expanded,L)
    gmmc.train(D,L)

    #Predicting the test set
    print("Predicting the test set...")
    _  = svmc.transform(DT)
    EvalScoresSVM = svmc.get_scores()
    _ = qlr.transform(expandedT)
    EvalScoresQLR = qlr.get_scores()
    _ = gmmc.transform(DT)
    EvalScoresGMM = gmmc.get_scores()

    EvalScoresSVM = np.reshape(EvalScoresSVM,(1,len(EvalScoresSVM)))
    EvalScoresQLR = np.reshape(EvalScoresQLR,(1,len(EvalScoresQLR)))
    EvalScoresGMM = np.reshape(EvalScoresGMM,(1,len(EvalScoresGMM)))

    #Calibrating Scores
    print("Calibrating Scores...")
    PredictedSVM = calibratorSVM.transform(EvalScoresSVM)
    CalibratedScoresSVM = calibratorSVM.get_scores()
    PredictedQLR = calibratorQLR.transform(EvalScoresQLR)
    CalibratedScoresQLR = calibratorQLR.get_scores()
    PredictedGMM = calibratorGMM.transform(EvalScoresGMM)
    CalibratedScoresGMM = calibratorGMM.get_scores()

    np.save("PredictedSVM.npy", PredictedSVM)
    np.save("PredictedQLR.npy", PredictedQLR)
    np.save("PredictedGMM.npy", PredictedGMM)
    np.save("CalibratedScoresSVM.npy", CalibratedScoresSVM)
    np.save("CalibratedScoresQLR.npy", CalibratedScoresQLR)
    np.save("CalibratedScoresGMM.npy", CalibratedScoresGMM)

    PredictedSVM = np.load("PredictedSVM.npy")
    PredictedQLR = np.load("PredictedQLR.npy")
    PredictedGMM = np.load("PredictedGMM.npy")
    CalibratedScoresSVM = np.load("CalibratedScoresSVM.npy")
    CalibratedScoresQLR = np.load("CalibratedScoresQLR.npy")
    CalibratedScoresGMM = np.load("CalibratedScoresGMM.npy")
   
    #Calculating minDCF
    print("Calculating minDCF...")
    minDCFSVM1 = val.min_DCF(CalibratedScoresSVM, 0.1, 1, 1, LT, PredictedSVM)
    minDCFSVM2 = val.min_DCF(CalibratedScoresSVM, 0.5, 1, 1, LT, PredictedSVM)
    minDCFQLR1 = val.min_DCF(CalibratedScoresQLR, 0.1, 1, 1, LT, PredictedQLR)
    minDCFQLR2 = val.min_DCF(CalibratedScoresQLR, 0.5, 1, 1, LT, PredictedQLR)
    minDCFGMM1 = val.min_DCF(CalibratedScoresGMM, 0.1, 1, 1, LT, PredictedGMM)
    minDCFGMM2 = val.min_DCF(CalibratedScoresGMM, 0.5, 1, 1, LT, PredictedGMM)

    #Calculating actDCF
    print("Calculating actDCF...")
    actualDCFSVM1 = val.act_DCF(CalibratedScoresSVM, 0.1, 1, 1, LT, None)
    actualDCFSVM2 = val.act_DCF(CalibratedScoresSVM, 0.5, 1, 1, LT, None)
    actualDCFQLR1 = val.act_DCF(CalibratedScoresQLR, 0.1, 1, 1, LT, None)
    actualDCFQLR2 = val.act_DCF(CalibratedScoresQLR, 0.5, 1, 1, LT, None)
    actualDCFGMM1 = val.act_DCF(CalibratedScoresGMM, 0.1, 1, 1, LT, None)
    actualDCFGMM2 = val.act_DCF(CalibratedScoresGMM, 0.5, 1, 1, LT, None)

    #Calculating minCprim
    print("Calculating minCprim...")
    minCprimSVM = (minDCFSVM1+minDCFSVM2)/2
    minCprimQLR = (minDCFQLR1+minDCFQLR2)/2
    minCprimGMM = (minDCFGMM1+minDCFGMM2)/2

    #Calculating Cprim
    print("Calculating Cprim...")
    CprimSVM = (actualDCFSVM1+actualDCFSVM2)/2
    CprimQLR = (actualDCFQLR1+actualDCFQLR2)/2
    CprimGMM = (actualDCFGMM1+actualDCFGMM2)/2

    #Printing results
    print("minDCF SVM 0.1:", minDCFSVM1)
    print("minDCF SVM 0.5:", minDCFSVM2)
    print("actDCF SVM 0.1:", actualDCFSVM1)
    print("actDCF SVM 0.5:", actualDCFSVM2)
    print("minCprim SVM:", minCprimSVM)
    print("Cprim SVM:", CprimSVM)

    print("minDCF QLR 0.1:", minDCFQLR1)
    print("minDCF QLR 0.5:", minDCFQLR2)
    print("actDCF QLR 0.1:", actualDCFQLR1)
    print("actDCF QLR 0.5:", actualDCFQLR2)
    print("minCprim QLR:", minCprimQLR)
    print("Cprim QLR:", CprimQLR)

    print("minDCF GMM 0.1:", minDCFGMM1)
    print("minDCF GMM 0.5:", minDCFGMM2)
    print("actDCF GMM 0.1:", actualDCFGMM1)
    print("actDCF GMM 0.5:", actualDCFGMM2)
    print("minCprim GMM:", minCprimGMM)
    print("Cprim GMM:", CprimGMM)
