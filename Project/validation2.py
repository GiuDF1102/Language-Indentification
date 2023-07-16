import numpy as np
import matplotlib.pyplot as plt
import data_utils as du
import gaussian_classifiers as gc
from sklearn.utils import shuffle
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import math_utils as mu
import SVM_classifiers as svm
import data_visualization as dv

class confusion_matrix:
    true_labels = []
    predicted_labels = []
    save = False
    num_classes = 0
    FNR = 0
    FPR = 0
    
    def __init__(self, true_labels, predicted_labels, save):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.save = save
        self.num_classes = len(np.unique(self.true_labels))
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def get_confusion_matrix(self):
        self.num_classes = len(np.unique(self.true_labels))
        joined_classes = np.array([self.true_labels, self.predicted_labels])
        for i in range(len(self.true_labels)):
            col = joined_classes[0][i]
            row = joined_classes[1][i]
            self.confusion_matrix[row][col] += 1
        return self.confusion_matrix
    
    def print_confusion_matrix(self, name):
        fig, ax = plt.subplots()
        
        ax.imshow(self.confusion_matrix, cmap='Blues')
    
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(np.arange(0, self.num_classes))
        ax.set_yticklabels(np.arange(0, self.num_classes))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, self.confusion_matrix[i, j], ha='center', va='center', color='black')
        
        plt.savefig("Confusion Matrix - {}.png".format(name))
        plt.close()
        
    
    def FNR_FPR_binary(self):
        cm = self.confusion_matrix
        FNR = cm[0][1]/(cm[0][1]+cm[1][1])
        FPR = 1 - FNR
        return (FNR, FPR)
    
    def DCF_binary(self,pi, C):
        FNR, FPR = self.FNR_FPR_binary()
        Cfn = C[0][1]
        Cfp = C[1][0]
        return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)

    def DCF_binary_norm(self,pi, C):
        FNR, FPR = self.FNR_FPR_binary()
        Cfn = C[0][1]
        Cfp = C[1][0]
        return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)/np.min([pi*Cfn, (1-pi)*Cfp])

def DCF_binary(pi, Cfn, Cfp, true_labels, predicted_labels):
    num_classes = len(np.unique(true_labels))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    joined_classes = np.array([true_labels, predicted_labels])
    for i in range(len(true_labels)):
        col = joined_classes[0][i]
        row = joined_classes[1][i]
        confusion_matrix[row][col] += 1
    FNR = confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])
    FPR = confusion_matrix[1][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)

def DCF(pi, Cfn, Cfp, true_labels, predicted_labels):
    DCF_u = DCF_binary(pi, Cfn, Cfp, true_labels, predicted_labels)
    DCFDummy = np.min([pi*Cfn, (1-pi)*Cfp])
    return DCF_u/DCFDummy
    
def FNR_FPR_binary_ind(confusion_matrix):
    cm = confusion_matrix
    FNR = cm[0][1]/(cm[0][1]+cm[1][1])
    FPR = cm[1][0]/(cm[0][0]+cm[1][0])
    return (FNR, FPR)
    
def min_DCF(scores, pi, Cfn, Cfp, true_labels, predicted_labels):
    sorted_scores = sorted(scores)
    min_dcf = np.inf
    best_threshold = None
    for t in sorted_scores:
        predicted_labels = np.where(scores>t,1,0)
        dcf = DCF(pi, Cfn, Cfp, true_labels, predicted_labels)
        if dcf < min_dcf:
            min_dcf = dcf
            best_threshold = t
    return min_dcf

def get_ROC(scores, true_labels, name):
    sorted_scores = sorted(scores)
    FPR_list = []
    TPR_list = []
    for t in sorted_scores:
        predicted_labels = np.where(scores>t,1,0)
        cnf_mat = confusion_matrix(true_labels, predicted_labels, False)
        cm = cnf_mat.get_confusion_matrix()
        FNR, FPR = FNR_FPR_binary_ind(cm)
        TPR = 1 - FNR
        FPR_list.append(FPR)
        TPR_list.append(TPR)
    
    plt.plot(FPR_list, TPR_list, linestyle='-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(True)
    plt.savefig("ROC - {}.png".format(name))
    plt.close()
        
def get_error_plot(scores, Cfn, Cfp, true_labels, predicted_labels, name):
    effPriorLogOdds = np.linspace(-3,3,21)
    pi = 1/(1+np.exp(-effPriorLogOdds))
    dcf = []
    min_dcf = []
    for p in pi:
        min_dcf.append(min_DCF(scores, p, Cfn, Cfp, true_labels, predicted_labels))
        dcf.append(DCF(p, Cfn, Cfp, true_labels, predicted_labels))    
    plt.plot(effPriorLogOdds, dcf, label='min DCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.ylabel('DCF value')
    plt.xlabel('prior log-odds')
    plt.savefig("error plot - {}.png".format(name))
    plt.close()
        

def binary_threshold(pi, C):
    Cfn = C[0][1]
    Cfp = C[1][0]
    t = - np.log((pi*Cfn)/(1-pi)*Cfp)
    return t

def k_fold_bayes_plot(learner,x,labels,k, workingPoint,name):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X, Y = shuffle(x.T, labels, random_state=0)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_scores = []
    concat_predicted = []
    for i in range(k): #for each fold
        print(f"FOLD {i}")
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train)
        concat_predicted.append(learner.transform(X_val))
        scores = learner.get_scores()
        concat_scores.append(scores)
    gotscores = np.hstack(concat_scores)
    gotpredicted = np.hstack(concat_predicted)
    threshold = -np.log(pi*Cfn) + np.log((1-pi)*Cfp)
    gotpredicted = np.where(gotscores>threshold,1,0)
    actualDCF = DCF(pi, Cfn, Cfp, Y, gotpredicted)
    minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    print(actualDCF, minDCF)
    #get_error_plot(gotscores, Cfn, Cfp, Y, gotpredicted, name)
    return actualDCF, minDCF, gotscores

def k_fold(learner,x,labels,k, workingPoint):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X, Y = shuffle(x.T, labels, random_state=0)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_scores = []
    concat_predicted = []
    for i in range(k): #for each fold
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train)
        concat_predicted.append(learner.transform(X_val))
        scores = learner.get_scores()
        concat_scores.append(scores)
    gotscores = np.hstack(concat_scores)
    gotpredicted = np.hstack(concat_predicted)
    threshold = -np.log(pi*Cfn) + np.log((1-pi)*Cfp)
    gotpredicted = np.where(gotscores>threshold,1,0)
    actualDCF = DCF(pi, Cfn, Cfp, Y, gotpredicted)
    minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    return actualDCF, minDCF

def k_fold_bayes_plot_calibrated(learner,x,labels,k, workingPoint,name):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X, Y = shuffle(x.T, labels, random_state=0)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_scores = []
    concat_predicted = []
    for i in range(k): #for each fold
        print(f"FOLD {i}")
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train)
        concat_predicted.append(learner.transform(X_val))
        scores = learner.get_scores()
        concat_scores.append(scores)
    gotscores = np.hstack(concat_scores)
    gotpredicted = np.hstack(concat_predicted)
    threshold = -np.log(pi*Cfn) + np.log((1-pi)*Cfp)
    gotscores = mu.FromColumnToRow(gotscores)
    
    lrc = lr.logReg(0,pi,"prova")
    lrc.train(gotscores,Y)
    alpha, beta = lrc.get_params()
    
    calibScores = alpha*gotscores+beta-np.log(pi/(1-pi))
    calibScores = calibScores[0]
    calibLabels = np.where(calibScores>threshold,1,0)
    actualDCF_ = DCF(pi, Cfn, Cfp, Y, calibLabels)
    minDCF_ = min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    print(actualDCF_, minDCF_)
    # get_error_plot(gotscores, Cfn, Cfp, Y, gotpredicted, name)
    return actualDCF_, minDCF_, gotscores


def split_db_2to1(D, L, seed=0):
    # 2/3 dei dati per il training----->100 per training, 50 per test
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  # DTR= data training, LTR= Label training
    # DTE= Data test, LTE= label testing


def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix

def compute_actual_DCF(llr, LTE, pi1, cfn, cfp):
    
    predictions = (llr > (-np.log(pi1/(1-pi1)))).astype(int)
    
    confMatrix =  confusionMatrix(predictions, LTE, LTE.max()+1)
    uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
    NDCF=(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    return NDCF

def detection_cost_function (M, pi1, cfn, cfp):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    
    return (pi1*cfn*FNR +(1-pi1)*cfp*FPR)

def normalized_detection_cost_function (DCF, pi1, cfn, cfp):
    dummy = np.array([pi1*cfn, (1-pi1)*cfp])
    index = np.argmin (dummy) 
    return DCF/dummy[index]

if __name__ == "__main__":

    # L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    # svmc = svm.SVM("RBF", True, gamma=0.01, C=0.1, K=0.01, piT=0.2)
    # DPCA = dr.PCA(D, 5)
    # actualDCF, minDCF, scores = k_fold_bayes_plot(svmc, DPCA, L, 5, (0.1, 1, 1), "NaivePCA5")
    # print(f"Not Calibrated, aDCF: {actualDCF}, minDCF: {minDCF}")
    # print("scores: ",scores)

    # actualDCF1, minDCF1, _ = k_fold_bayes_plot_calibrated(svmc, DPCA, L, 5, (0.1, 1, 1), "NaivePCA5Calibrated")  
    # print(f"Calibrated, aDCF: {actualDCF1}, minDCF: {minDCF1}")  


    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")    
    lambdas = np.logspace(-2, 5, num=30)
    k = 5

    featuresZNorm = mu.z_score(features)
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTrainQuadraticZNorm = du.features_expansion(featuresZNorm)
    featuresPCA5 = dr.PCA(features,5)
    featuresPCA4 = dr.PCA(features,4)
    featuresPCA5ZNorm = mu.z_score(featuresPCA5)
    featuresPCA5Exapanded = du.features_expansion(featuresPCA5)
    featuresPCA4Exapanded = du.features_expansion(featuresPCA4)
    featuresPCA5ZNormExapanded = du.features_expansion(dr.PCA(mu.z_score(features), 5))

    ##LINEAR NOT NORM + LINEAR Z-NORM EMPIRICAL
    labels_text = [
        "Linear LogReg",
        "Linear LogReg Z-norm"
    ]
    CprimLogReg = np.zeros((2, len(lambdas)))
    piT = 0.17
    with open("LinearLogRegEmpiricalNoNorm_Norm.txt", "w") as f:
        ##LINEAR NOT NORMALIZED piT = empirical
        minDCFList = np.zeros((2, len(lambdas)))
        for index, piTilde in enumerate([0.1,0.5]):
            for lIndex, l in enumerate(lambdas):
                logRegObj = lrc.logReg(l, piT, "balanced") 
                _, minDCF = k_fold(logRegObj, features, labels, k, (piTilde, 1,1))
                print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
                print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
                minDCFList[index, lIndex] = minDCF
        CprimLogReg[0] = minDCFList.mean(axis=0)
        print("Cprim list {}".format(CprimLogReg[0]), file=f)
        print("Cprim list {}".format(CprimLogReg[0]))


        ##LINEAR NORMALIZED piT = empirical
        minDCFList = np.zeros((2, len(lambdas)))
        for index, piTilde in enumerate([0.1,0.5]):
            for lIndex, l in enumerate(lambdas):
                logRegObj = lrc.logReg(l, piT, "balanced")
                _, minDCF = k_fold(logRegObj, featuresZNorm, labels, k, (piTilde, 1, 1))
                print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
                print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
                minDCFList[index, lIndex] = minDCF
        CprimLogReg[1] = minDCFList.mean(axis=0)
        print("Cprim list {}".format(CprimLogReg[1]), file=f)
        print("Cprim list {}".format(CprimLogReg[1]))

        dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalNoNorm_Norm")

    # ##LINEAR NOT NORM PCAs EMPIRICAL
    # labels_text = [
    #     "Linear LogReg NO PCA",
    #     "Linear LogReg PCA 5",
    #     "Linear LogReg PCA 4"
    # ]
    # CprimLogReg = np.zeros((3, len(lambdas)))
    # piT = 0.17
    # with open("LinearLogRegEmpiricalPCA.txt", "w") as f:
    #     ##LINEAR NOT NORMALIZED piT = empirical No PCA
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, features, labels, k, (piTilde, 1,1))
    #             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[0] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[0]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[0]))

    #     ##LINEAR NOT NORMALIZED piT = empirical PCA 5
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA5, labels, k, (piTilde, 1,1))
    #             print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[1] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[1]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[1]))

    #     ##LINEAR NOT NORMALIZED piT = empirical PCA 4
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA4, labels, k, (piTilde, 1,1))
    #             print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[2] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[2]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[2]))

    #     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalPCA")

    # #----------------------------------------------
    # QUADRATIC
    # labels_text = [
    #     "Quadratic LogReg",
    #     "Quadratic LogReg Z-norm"
    # ]
    # piT = 0.17
    # with open("QLogRegEmpiricalNoNorm_Norm.txt", "w") as f:
    #     CprimLogReg = np.zeros((2, len(lambdas)))
    #     #QUADRATIC NOT NORMALIZED piT = empirical
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[0] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg), file=f)
    #     print("Cprim list {}".format(CprimLogReg))

    #     #QUADRATIC NORMALIZED piT = empirical
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced")
    #             _, minDCF = k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1, 1))
    #             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF), file=f)
    #             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[1] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg), file=f)
    #     print("Cprim list {}".format(CprimLogReg))

    #     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QLogRegEmpiricalNoNorm_Norm")


    # #QUADRATIC NOT NORM PCAs EMPIRICAL
    # labels_text = [
    #     "Quadratic LogReg NO PCA",
    #     "Quadratic LogReg NO PCA Z-Norm",
    #     "Quadratic LogReg PCA 5",
    #     "Quadratic LogReg PCA 5 Z-Norm",
    # ]
    # CprimLogReg = np.zeros((4, len(lambdas)))
    # piT = 0.17
    # with open("QuadraticLogRegEmpiricalPCANorm.txt", "w") as f:
    #     QUADRATIC NOT NORMALIZED piT = empirical No PCA
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[0] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[0]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[0]))

    #     QUADRATIC NORMALIZED piT = empirical No PCA
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[1] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[1]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[1]))

    #     QUADRATIC NOT NORMALIZED piT = empirical PCA 5
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[2] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[2]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[2]))

    #     QUADRATIC NORMALIZED piT = empirical PCA 5
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA5ZNormExapanded, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[3] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[3]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[3]))

    #     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCANorm")

    #---------------------------------
    ##QUADRATIC NOT NORM PCAs EMPIRICAL
    # labels_text = [
    #     "Quadratic LogReg NO PCA",
    #     "Quadratic LogReg PCA 5",
    #     "Quadratic LogReg PCA 4"
    # ]
    # CprimLogReg = np.zeros((3, len(lambdas)))
    # piT = 0.17
    # with open("QuadraticLogRegEmpiricalPCATEST2.txt", "w") as f:
    #     #QUADRATIC NOT NORMALIZED piT = empirical No PCA
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[0] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[0]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[0]))

    #     #QUADRATIC NOT NORMALIZED piT = empirical PCA 5
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[1] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[1]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[1]))

    #     #QUADRATIC NOT NORMALIZED piT = empirical PCA 4
    #     minDCFList = np.zeros((2, len(lambdas)))
    #     for index, piTilde in enumerate([0.1,0.5]):
    #         for lIndex, l in enumerate(lambdas):
    #             logRegObj = lrc.logReg(l, piT, "balanced") 
    #             _, minDCF = k_fold(logRegObj, featuresPCA4Exapanded, labels, k, (piTilde, 1,1))
    #             print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
    #             print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
    #             minDCFList[index, lIndex] = minDCF
    #     CprimLogReg[2] = minDCFList.mean(axis=0)
    #     print("Cprim list {}".format(CprimLogReg[2]), file=f)
    #     print("Cprim list {}".format(CprimLogReg[2]))

    #     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCATEST2")

    # with open("QLOGREG_PI_PITILDE.txt","w") as f:
    #     for piT in [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20]:
    #         CprimList=[]
    #         for piTilde in [0.1,0.5]:
    #             logRegObj = lrc.logReg(lambdaBest, piT, "balanced")
    #             _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (piTilde, 1, 1))
    #             print("Quadratic LogReg Balanced Not Normalized, minDCF with piT {} and piTilde {} no PCA and  is {}".format(piT, piTilde, minDCF),file=f)
    #             CprimList.append(minDCF)
    #         Cprim=np.array(CprimList).mean(axis=0)
    #         print("Quadratic LogReg Balanced Not Normalized, Cprim with piT {}  no PCA and  is {}".format(piT,Cprim ),file=f)
