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
import validation2 as val
import validation as val1

def k_fold_fusion(learner,x,labels,k, workingPoint):
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
    #actualDCF = act_DCF(pi, Cfn, Cfp, Y, gotpredicted)
    #minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    return gotscores

def k_fold_fusionV2(learner,x,labels,k, workingPoint):
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
    actualDCF= val.act_DCF(gotscores,pi, Cfn, Cfp, Y, None)
    minDCF = val.min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    return actualDCF,actualDCF,gotscores

def k_fold_bayes_plot_calibrated(learner,x,labels,k, workingPoint):
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
    actualDCF_ = val.act_DCF(calibScores,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    print(actualDCF_, minDCF_)
    #get_error_plot(calibScores, Cfn, Cfp, Y, calibLabels, name)
    return actualDCF_, minDCF_, gotscores

def k_fold_fusionV3(learner1, learner2,x,labels,k, workingPoint):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X, Y = shuffle(x.T, labels, random_state=0)
    X_splitted = np.array_split(X, k)
    y_splitted = np.array_split(Y, k)
    concat_scores1 = []
    concat_scores2 = []
    concat_predicted1 = []
    concat_predicted2 = []
    for i in range(k): #for each fold
        X_folds = X_splitted.copy()
        y_folds = y_splitted.copy()
        X_val = X_folds.pop(i).T
        y_val = y_folds.pop(i)
        X_train = np.vstack(X_folds).T
        y_train = np.hstack(y_folds)
        learner1.train(X_train, y_train)
        learner2.train(X_train, y_train)
        scores1 = learner1.get_scores()
        scores2 = learner2.get_scores()
        concat_predicted1.append(learner1.transform(X_val))
        concat_predicted2.append(learner2.transform(X_val))
        concat_scores1.append(scores1)
        concat_scores2.append(scores2)
    gotscores1 = np.hstack(concat_scores1)
    gotscores2 = np.hstack(concat_scores2)
    gotscores = np.hstack(concat_scores)
    gotpredicted = np.hstack(concat_predicted)
    gotscores = mu.FromColumnToRow(gotscores)
    
    lrcobj = lrc.logReg(0,pi,"balanced")
    lrcobj.train(gotscores,Y)
    alpha, beta = lrcobj.get_params()
    
    calibScores = alpha*gotscores+beta-np.log(pi/(1-pi))
    calibScores = calibScores[0]
    calibLabels = np.where(calibScores>0,1,0)
    actualDCF_ = val.act_DCF(calibScores,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    print(actualDCF_, minDCF_)
    #get_error_plot(calibScores, Cfn, Cfp, Y, calibLabels, name)
    return actualDCF_, minDCF_, gotscores

if __name__=="__main__":
    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    PCA5=dr.PCA(D,5)
    bestLambda=10
    k=2
    lr=lrc.logReg(bestLambda,0.17,"balanced")
    DPCA = du.features_expansion(D)
    scores1=k_fold_fusion(lr,DPCA,L,k,(0.1,1,1))

    svmc=svm.SVM("RBF",True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    scores2=k_fold_fusion(svmc,PCA5,L,k,(0.1,1,1))

    scoresToLog=np.vstack((scores1,scores2))


    L = shuffle(L, random_state=0)

    slr = lrc.logReg(0,0.1,"balanced")
    slr.train(scoresToLog,L)
    w,b=slr.get_params()
    f_s = np.dot(w.T,scoresToLog)+b

    lab = np.where(f_s>0,1,0)

    actDCF1 = val.act_DCF(f_s,0.1,1,1,L,None)
    minDCF1 = val.min_DCF(f_s, 0.1, 1, 1, L, lab)

    print("-----------------prior 0.1")
    print(actDCF1, minDCF1)
#-------------------------------------

    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    PCA5=dr.PCA(D,5)
    bestLambda=10
    k=2
    lr=lrc.logReg(bestLambda,0.17,"balanced")
    DPCA = du.features_expansion(D)
    scores1=k_fold_fusion(lr,DPCA,L,k,(0.5,1,1))

    svmc=svm.SVM("RBF",True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    scores2=k_fold_fusion(svmc,PCA5,L,k,(0.5,1,1))

    scoresToLog=np.vstack((scores1,scores2))

    L = shuffle(L, random_state=0)

    slr = lrc.logReg(0,0.5,"balanced")
    slr.train(scoresToLog,L)
    w,b=slr.get_params()
    f_s = np.dot(w.T,scoresToLog)+b

    lab = np.where(f_s>0,1,0)

    actDCF2 = val.act_DCF(f_s,0.5,1,1,L,None)
    minDCF2 = val.min_DCF(f_s, 0.5, 1, 1, L, lab)

    print("-----------------prior 0.5")
    print(actDCF2, minDCF2)

    Cprim = (actDCF1+actDCF2)/2
    minCprim = (minDCF1+minDCF2)/2

    print("----------------- Cprim")
    print(Cprim, minCprim)

