import numpy as np
import data_utils as du
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def calcErrorRate(lab, pre):
    confronted = lab == pre
    correct = np.sum(np.where(confronted == True, 1, 0))
    return (1-correct/len(lab))*100

def calc_accuracy(labels, predicted):
    confronted = (labels == predicted)
    TP = 0
    
    for i in confronted:
        if(i == True):
            TP = TP + 1
    
    return TP/len(predicted)

def k_fold_v1(learners,x,labels,k):
    error_rates = []
    for learner in learners:
        X, Y = shuffle(x.T, labels)
        X = np.array_split(X, k)
        y = np.array_split(Y, k)
        concat_predicted = []
        for i in range(k): #for each fold
            X_train = np.concatenate(np.delete(X, i, axis=0), axis=0).T
            y_train = np.concatenate(np.delete(y, i, axis=0), axis=0)
            X_val = X[i].T
            learner.fit(X_train, y_train)
            predicted = learner.trasform(X_val)
            concat_predicted.extend((predicted.tolist()))
        error_rates.append((1-calc_accuracy(Y, np.array(concat_predicted)))*100)
    return error_rates

def k_fold_SVM_linear(learner,x,labels,k, C, K):
    X, Y = shuffle(x.T, labels)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_predicted = []
    for i in range(k): #for each fold
        print("SVM Linear Fold: ", i+1)
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train, C=C, K=K)
        predicted = learner.trasform(X_val)
        concat_predicted.extend((predicted.tolist()))
    conf_matr = confusion_matrix(Y, np.array(concat_predicted))
    return conf_matr

def k_fold_SVM_RBF(learner,x,labels,k, C, K, gamma):
    X, Y = shuffle(x.T, labels)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_predicted = []
    for i in range(k): #for each fold
        print("SVM RBF Fold: ", i+1)
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train, C=C, K=K, gamma=gamma)
        predicted = learner.trasform(X_val)
        concat_predicted.extend((predicted.tolist()))
    conf_matr = confusion_matrix(Y, np.array(concat_predicted))
    return conf_matr

def k_fold_SVM_Polinomial(learner,x,labels,k, C, K, d,c):
    X, Y = shuffle(x.T, labels)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_predicted = []
    for i in range(k): #for each fold
        print("SVM Polinomial Fold: ", i+1)
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train, C=C, K=K, d=d, c=c)
        predicted = learner.trasform(X_val)
        concat_predicted.extend((predicted.tolist()))
    conf_matr = confusion_matrix(Y, np.array(concat_predicted))
    return conf_matr

class confusion_matrix:
    true_labels = []
    predicted_labels = []
    num_classes = 0
    FNR = 0
    FPR = 0
    
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
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
        FPR = cm[1][0]/(cm[0][0]+cm[1][0])
        return (FNR, FPR)
    
    def DCF_binary(self,pi, C):
        FNR, FPR = self.FNR_FPR_binary()
        Cfn = C[0][1]
        Cfp = C[1][0]
        return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)

        #usign a dummy bayesian for reference: min:(pi*Cfn,(1-pi)*Cfp)
    def DCF_binary_norm(self,pi, C):
        DCFu=self.DCF_binary(pi,C)
        return (DCFu)/np.minimum(pi*C[0][1], (1-pi)*C[1][0])

def FNR_FPR_binary_ind(confusion_matrix):
    cm = confusion_matrix
    FNR = cm[0][1]/(cm[0][1]+cm[1][1])
    FPR = cm[1][0]/(cm[0][0]+cm[1][0])
    return (FNR, FPR)

def DCF_binary_norm_ind(cm, pi,C):
    TP = cm[0,0]
    FN = cm[1,0]
    FP = cm[0,1]
    TN = cm[1,1]
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    Cfn = C[0][1]
    Cfp = C[1][0]
    return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)/np.min([pi*Cfn, (1-pi)*Cfp])
    
def min_DCF(scores, true_labels, pi, C):
    sorted_scores = sorted(scores)
    min_dcf = np.inf
    best_threshold = None
    for t in sorted_scores:
        predicted_labels = np.where(scores>t,1,0)
        cnf_mat = confusion_matrix(true_labels, predicted_labels)
        M = cnf_mat.get_confusion_matrix()
        normDCF = calc_normDCF_v2(M, pi, C)
        if normDCF < min_dcf:
            min_dcf = normDCF
            best_threshold = t
    return min_dcf, best_threshold

def calc_normDCF_v2(M,pi,C):
    cfn = C[0][1]
    cfp = C[1][0]
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    DCF = (pi*cfn*FNR +(1-pi)*cfp*FPR)
    dummy = np.array([pi*cfn, (1-pi)*cfp])
    index = np.argmin(dummy) 
    normDCF = DCF/dummy[index]
    return normDCF

def DCF(scores, true_labels, pi, C):
    Cfn = C[0][1]
    Cfp = C[1][0]
    t = - np.log((pi*Cfn)/(1-pi)*Cfp)
    predicted_labels = np.where(scores>t,1,0)
    cnf_mat = confusion_matrix(true_labels, predicted_labels)
    cm = cnf_mat.get_confusion_matrix()
    FNR, FPR = FNR_FPR_binary_ind(cm)
    return (pi*Cfn*FNR+(1-pi)*Cfp*FPR)/np.min([pi*Cfn, (1-pi)*Cfp]) 

def get_ROC(scores, true_labels, pi, C, name):
    sorted_scores = sorted(scores)
    FPR_list = []
    TPR_list = []
    for t in sorted_scores:
        predicted_labels = np.where(scores>t,1,0)
        cnf_mat = confusion_matrix(true_labels, predicted_labels)
        cm = cnf_mat.get_confusion_matrix()
        FNR, FPR = FNR_FPR_binary_ind(cm)
        TPR = 1 - FNR
        FPR_list.append(FPR)
        TPR_list.append(TPR)
    
    plt.plot(FPR_list, TPR_list, linestyle='-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(True)
    plt.title("ROC - {}".format(name))
    plt.savefig("ROC - {}.png".format(name))
    plt.close()
        
def get_error_plot(scores, true_labels, C,name):
    effPriorLogOdds = np.linspace(-3,3,21)
    pi = 1/(1+np.exp(-effPriorLogOdds))
    dcf = []
    min_dcf = []
    for p in pi:
        min_dcf.append(min_DCF(scores, true_labels, p, C))
        dcf.append(DCF(scores,true_labels, p, C))   
        
    plt.plot(effPriorLogOdds, dcf, label='min DCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.ylabel('DCF value')
    plt.xlabel('prior log-odds')
    plt.title("ROC - {}".format(name))
    plt.savefig("error plot - {}.png".format(name))
    plt.close()

def binary_threshold(pi, C):
    Cfn = C[0][1]
    Cfp = C[1][0]
    t = - np.log((pi*Cfn)/(1-pi)*Cfp)
    return t

def optimal_bayes_decision_binary(scores, labels, workingPoint):
    # workingPoint (pi, Cfn, Cfp)
    prior = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    Cx0 = Cfn*scores[1]
    Cx1 = Cfp*scores[0]
    c = np.vstack((Cx0,Cx1))
    c_min = np.argmin(c, axis=0)
    cnf_matr = confusion_matrix(labels, c_min)
    cm = cnf_matr.get_confusion_matrix()
    return cm

def optimal_bayes_decision_binary_t(scores, labels, workingPoint,t):
    # workingPoint (pi, Cfn, Cfp)
    prior = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    Cx0 = Cfn*scores[1]
    Cx1 = Cfp*scores[0]
    r = np.log(Cx0/Cx1)
    labels_r = np.where(r>t,1,0)
    cnf_matr = confusion_matrix(labels, labels_r)
    cm = cnf_matr.get_confusion_matrix()
    return cm

def compute_bayes_risk(confusion_matrix, workingPoint):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    FNR = confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])
    FPR = confusion_matrix[1][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    DCFu = (pi*Cfn*FNR +(1-pi)*Cfp*FPR)
    return DCFu

def compute_dummy_bayes(workingPoint):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    dummy = np.array([pi*Cfn, (1-pi)*Cfp])
    return np.min(dummy) 

def compute_minDCF(scores, labels, workingPoint):
    llr = np.log(scores[0])-np.log(scores[1])
    sorted_scores = sorted(llr)
    min_dcf = np.inf
    best_threshold = None
    for t in sorted_scores:
        cm = optimal_bayes_decision_binary_t(scores, labels, workingPoint,t)
        DCFu = compute_bayes_risk(cm, workingPoint)
        actualDCF = DCFu/compute_dummy_bayes(workingPoint)
        if actualDCF < min_dcf:
            min_dcf = actualDCF
            best_threshold = t
    return min_dcf, best_threshold

def k_fold(learner,x,labels,k, workingPoint, name):
    X, Y = shuffle(x.T, labels, random_state=0)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_predicted = []
    concat_scores = []
    concat_llr = []
    for i in range(k): #for each fold
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner.train(X_train, y_train)
        learner.transform(X_val)
        scores = learner.get_scores()
        concat_scores.append(scores)
    gotscores = np.hstack(concat_scores)
    cm = optimal_bayes_decision_binary(gotscores, Y, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF = DCFu/compute_dummy_bayes(workingPoint)
    minDCF, _ = compute_minDCF(gotscores, Y, workingPoint)
    print("{} minDCF: {}".format(name, minDCF))