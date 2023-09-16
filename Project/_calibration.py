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
    
    svmc = svm.SVM("RBF", balanced=True, K=0.01, C=0.1, gamma=0.01, piT=0.2)
    DPCA5 = dr.PCA(D, 5)
    actualDCFSVM1, minDCFSVM1, scoresSVM1, predictedSVM1 = val.k_fold_bayes_plot(svmc, DPCA5, L, 5, (0.1, 1, 1), "PROVASVM", False)
    actualDCFSVM2, minDCFSVM2, scoresSVM2, predictedSVM2 = val.k_fold_bayes_plot(svmc, DPCA5, L, 5, (0.5, 1, 1), "PROVASVM", False)
    print("minDCF SVM 0.1:", minDCFSVM1)
    print("minDCF SVM 0.5:", minDCFSVM2)
    print("actDCF SVM 0.1:", actualDCFSVM1)
    print("actDCF SVM 0.5:", actualDCFSVM2)
    print("minCprim SVM:", (minDCFSVM1+minDCFSVM2)/2)
    print("Cprim SVM:", (actualDCFSVM1+actualDCFSVM2)/2)
    np.save("scoresSVM1.npy", scoresSVM1)
    np.save("scoresSVM2.npy", scoresSVM2)
    np.save("predictedSVM1.npy", predictedSVM1)
    np.save("predictedSVM2.npy", predictedSVM2)

    gmmc = gmm.GMM(2,32,"MVG", "tied")
    DPCA5 = dr.PCA(D, 5)
    actualDCFGMM1, minDCFGMM1, scoresGMM1,predictedGMM1 = val.k_fold_bayes_plot(gmmc, D, L, 5, (0.1, 1, 1), "PROVASVM", False)
    actualDCFGMM2, minDCFGMM2, scoresGMM2,predictedGMM2 = val.k_fold_bayes_plot(gmmc, D, L, 5, (0.5, 1, 1), "PROVASVM", False)
    print("minDCF GMM 0.1:", minDCFGMM1)
    print("minDCF GMM 0.5:", minDCFGMM2)
    print("actDCF GMM 0.1:", actualDCFGMM1)
    print("actDCF GMM 0.5:", actualDCFGMM2)
    print("minCprim GMM:", (minDCFGMM1+minDCFGMM2)/2)
    print("Cprim GMM:", (actualDCFGMM1+actualDCFGMM2)/2)
    np.save("scoresGMM1.npy", scoresGMM1)
    np.save("scoresGMM2.npy", scoresGMM2)
    np.save("predictedGMM1.npy", predictedGMM1)
    np.save("predictedGMM2.npy", predictedGMM2)
    
    qlogreg = lrc.logReg(10,0.1,"balanced")
    expanded = du.features_expansion(D)
    actualDCFQLR1, minDCFQLR1, scoresQLR1,predictedQLOGREG1 = val.k_fold_bayes_plot(qlogreg, expanded, L, 5, (0.1, 1, 1),"DETQLOGREG", False)
    actualDCFQLR2, minDCFQLR2, scoresQLR2,predictedQLOGREG2 = val.k_fold_bayes_plot(qlogreg, expanded, L, 5, (0.5, 1, 1),"DETQLOGREG", False)
    print("minDCF QLR 0.1:", minDCFQLR1)
    print("minDCF QLR 0.5:", minDCFQLR2)
    print("actDCF QLR 0.1:", actualDCFQLR1)
    print("actDCF QLR 0.5:", actualDCFQLR2)
    print("minCprim QLR:", (minDCFQLR1+minDCFQLR2)/2)
    print("Cprim QLR:", (actualDCFQLR1+actualDCFQLR2)/2)
    np.save("scoresQLR1.npy", scoresQLR1)
    np.save("scoresQLR2.npy", scoresQLR2)
    np.save("predictedQLR1.npy", predictedQLOGREG1)
    np.save("predictedQLR2.npy", predictedQLOGREG2)


    L_ = shuffle(L,random_state=0)
    legend = ["SVM actDCF", "SVM minDCF", "GMM actDCF", "GMM minDCF", "QLR actDCF", "QLR minDCF"]

    scoresSVM1 = np.load("scoresSVM1.npy")
    scoresSVM2 = np.load("scoresSVM2.npy")
    scoresGMM1 = np.load("scoresGMM1.npy")
    scoresGMM2 = np.load("scoresGMM2.npy")
    scoresQLR1 = np.load("scoresQLR1.npy")
    scoresQLR2 = np.load("scoresQLR2.npy")

    predictedSVM1 = np.load("predictedSVM1.npy")
    predictedSVM2 = np.load("predictedSVM2.npy")
    predictedGMM1 = np.load("predictedGMM1.npy")
    predictedGMM2 = np.load("predictedGMM2.npy")
    predictedQLR1 = np.load("predictedQLR1.npy")
    predictedQLR2 = np.load("predictedQLR2.npy")



    val.get_multi_error_plot([scoresSVM1, scoresGMM1, scoresQLR1], 1, 1, L_, [predictedGMM1, predictedGMM1, predictedQLR1], legend, "Error_Plots")


    lrcobj = lrc.logReg(0,0.5,"balanced")
    L = shuffle(L,random_state=0)
    scoresSVM1 = np.reshape(scoresSVM1, (1,scoresSVM1.shape[0]))
    scoresSVM2 = np.reshape(scoresSVM2, (1,scoresSVM2.shape[0]))
    scoresGMM1 = np.reshape(scoresGMM1, (1,scoresGMM1.shape[0]))
    scoresGMM2 = np.reshape(scoresGMM2, (1,scoresGMM2.shape[0]))
    scoresQLR1 = np.reshape(scoresQLR1, (1,scoresQLR1.shape[0]))
    scoresQLR2 = np.reshape(scoresQLR2, (1,scoresQLR2.shape[0]))


    # #CAL
    actualDCFSVM1, minDCFSVM1, scoresSVM1, predicetedSVM1 = val.k_fold_bayes_plot(lrcobj, scoresSVM1, L, 5, (0.1, 1, 1), "PROVASVM", None)
    actualDCFSVM2, minDCFSVM2, scoresSVM2, predicetedSVM2 = val.k_fold_bayes_plot(lrcobj, scoresSVM2, L, 5, (0.5, 1, 1), "PROVASVM", None)
    print("CALIBRATED minDCF 0.1 SVM:", minDCFSVM1)
    print("CALIBRATED minDCF 0.5 SVM:", minDCFSVM2)
    print("CALIBRATED minCprim SVM:", (minDCFSVM1+minDCFSVM2)/2)
    print("CALIBRATED actDCF 0.1 SVM:", actualDCFSVM1)
    print("CALIBRATED actDCF 0.5 SVM:", actualDCFSVM2)
    print("CALIBRATED Cprim SVM:", (actualDCFSVM1+actualDCFSVM2)/2)
    actualDCFGMM1, minDCFGMM1, scoresGMM1, predictedGMM1 = val.k_fold_bayes_plot(lrcobj, scoresGMM1, L, 5, (0.1, 1, 1), "PROVASVM", None)
    actualDCFGMM2, minDCFGMM2, scoresGMM2, predictedGMM2 = val.k_fold_bayes_plot(lrcobj, scoresGMM2, L, 5, (0.5, 1, 1), "PROVASVM", None)
    print("CALIBRATED minDCF 0.1 GMM:", minDCFGMM1)
    print("CALIBRATED minDCF 0.5 GMM:", minDCFGMM2)
    print("CALIBRATED actDCF 0.1 GMM:", actualDCFGMM1)
    print("CALIBRATED actDCF 0.5 GMM:", actualDCFGMM2)
    print("CALIBRATED minCprim GMM:", (minDCFGMM1+minDCFGMM2)/2)
    print("CALIBRATED Cprim GMM:", (actualDCFGMM1+actualDCFGMM2)/2)
    actualDCFQLR1, minDCFQLR1, scoresQLR1, predictedQLOGREG1 = val.k_fold_bayes_plot(lrcobj, scoresQLR1, L, 5, (0.1, 1, 1), "PROVASVM", None)
    actualDCFQLR2, minDCFQLR2, scoresQLR2, predictedQLOGREG2 = val.k_fold_bayes_plot(lrcobj, scoresQLR2, L, 5, (0.5, 1, 1), "PROVASVM", None)
    print("CALIBRATED minDCF 0.1 QLR:", minDCFQLR1)
    print("CALIBRATED minDCF 0.5 QLR:", minDCFQLR2)
    print("CALIBRATED actDCF 0.1 QLR:", actualDCFQLR1)
    print("CALIBRATED actDCF 0.5 QLR:", actualDCFQLR2)
    print("CALIBRATED minCprim QLR:", (minDCFQLR1+minDCFQLR2)/2)
    print("CALIBRATED Cprim QLR:", (actualDCFQLR1+actualDCFQLR2)/2)

    L_ = shuffle(L,random_state=0)

    val.get_multi_error_plot([scoresSVM1, scoresGMM1, scoresQLR1], 1, 1, L_, [predictedGMM1, predictedGMM1, predictedQLOGREG1], legend, "Error_Plots_Calibrated_Val")
