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
import GMM as gmm

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
        dcf = act_DCF(scores, pi, Cfn, Cfp, true_labels, t)
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
        dcf.append(act_DCF(scores, p, Cfn, Cfp, true_labels,None))    
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

def act_DCF(scores, pi, Cfn, Cfp, true_labels,t):
    
    if t is None:
        t = - np.log((pi*Cfn)/(1-pi)*Cfp)
    predicted_labels = np.where(scores>t,1,0)
    return DCF(pi, Cfn, Cfp, true_labels, predicted_labels)

#----------------------------------------------------


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
    actualDCF = act_DCF(gotscores, pi, Cfn, Cfp, Y, None)
    minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    print(actualDCF, minDCF)
    #get_error_plot(gotscores, Cfn, Cfp, Y, gotpredicted, name)
    return actualDCF, minDCF, gotscores, gotpredicted

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
    gotpredicted = np.where(gotscores>0,1,0)
    actualDCF = DCF(pi, Cfn, Cfp, Y, gotpredicted)
    minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    return actualDCF, minDCF, gotscores

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
    gotscores = mu.FromColumnToRow(gotscores)
    
    lrcobj = lrc.logReg(0,pi,"balanced")
    lrcobj.train(gotscores,Y)
    alpha, beta = lrcobj.get_params()
    
    calibScores = alpha*gotscores+beta-np.log(pi/(1-pi))
    calibScores = calibScores[0]
    calibLabels = np.where(calibScores>0,1,0)
    actualDCF_ = act_DCF(calibScores,pi, Cfn, Cfp, Y, None)
    minDCF_ = min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    print(actualDCF_, minDCF_)
    #get_error_plot(calibScores, Cfn, Cfp, Y, calibLabels, name)
    return actualDCF_, minDCF_, gotscores


if __name__ == "__main__":

    # L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    # svmc = gmm.GMM(2,32,"MVG", "tied")
    # DPCA = D
    # actualDCF, minDCF, scores = k_fold_bayes_plot(svmc, DPCA, L, 5, (0.1, 1, 1), "PROVAGMM")
    # print(f"Not Calibrated, aDCF: {actualDCF}, minDCF: {minDCF}")
    # print("scores: ",scores)

    # actualDCF1, minDCF1, _ = k_fold_bayes_plot_calibrated(svmc, DPCA, L, 5, (0.1, 1, 1), "PROVAGMMCALIBRATED")  
    # print(f"Calibrated, aDCF: {actualDCF1}, minDCF: {minDCF1}")  

    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    svmc = svm.SVM("RBF", balanced=True, K=0.1, C=0.01, gamma=0.01, piT=0.2)
    DPCA5 = dr.PCA(D, 5)
    actualDCF, minDCF, scoresSVM, lab = k_fold_bayes_plot(svmc, DPCA5, L, 5, (0.1, 1, 1), "PROVASVM")
    L = shuffle(L, random_state=0)
    actualDCF = act_DCF(scoresSVM,0.1,1,1,L,None)
    minDCF = min_DCF(scoresSVM, 0.1, 1, 1, L, lab)
    print(f"SVM aDCF: {actualDCF}, minDCF: {minDCF}")

    gmmc = gmm.GMM(2,32,"MVG", "tied")
    DPCA5 = dr.PCA(D, 5)
    actualDCF, minDCF, scores = k_fold_bayes_plot(gmmc, D, L, 5, (0.1, 1, 1), "PROVASVM")
    print(f"GMM aDCF: {actualDCF}, minDCF: {minDCF}")

    qlogreg = lrc.logReg(10,0.17,"balanced")
    expanded = du.features_expansion(D)
    actualDCF, minDCF, scores = k_fold_bayes_plot(qlogreg, expanded, L, 5, (0.1, 1, 1), "PROVASVM")
    print(f"QLOG aDCF: {actualDCF}, minDCF: {minDCF}")

    #CAL



    svmc = svm.SVM("RBF", balanced=True, K=0.1, C=0.01, gamma=0.01, piT=0.2)
    DPCA5 = dr.PCA(D, 5)
    actualDCF, minDCF, scores = k_fold_bayes_plot_calibrated(svmc, DPCA5, L, 5, (0.1, 1, 1), "PROVASVM")
    print(f"SVM CAL aDCF: {actualDCF}, minDCF: {minDCF}")

    gmmc = gmm.GMM(2,32,"MVG", "tied")
    DPCA5 = dr.PCA(D, 5)
    actualDCF, minDCF, scores = k_fold_bayes_plot_calibrated(gmmc, D, L, 5, (0.1, 1, 1), "PROVASVM")
    print(f"GMM CAL aDCF: {actualDCF}, minDCF: {minDCF}")

    qlogreg = lrc.logReg(10,0.17,"balanced")
    expanded = du.features_expansion(D)
    actualDCF, minDCF, scores = k_fold_bayes_plot_calibrated(qlogreg, expanded, L, 5, (0.1, 1, 1), "PROVASVM")
    print(f"QLOG CAL aDCF: {actualDCF}, minDCF: {minDCF}")