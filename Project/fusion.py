import numpy as np
import logistic_regression_classifiers as lrc
import SVM_classifiers as svm
import validation2 as val
import dimensionality_reduction as dr
import data_utils as du


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
    actualDCF= act_DCF(gotscores,pi, Cfn, Cfp, Y, None)
    minDCF = min_DCF(gotscores, pi, Cfn, Cfp, Y, gotpredicted)
    return actualDCF,actualDCF,gotscores



if __name__=="__main__":
    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    PCA5=dr.PCA(D,5)
    bestLambda=10
    k=2
    lr=lrc.logReg(bestLambda,0.1,"balanced")
    DPCA = du.features_expansion(D)
    scores1=k_fold_fusion(lr,DPCA,L,k,(0.1,1,1))
    print(scores1)
    svmc=svm.SVM("RBF",True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    scores2=k_fold_fusion(svmc,PCA5,L,k,(0.1,1,1))

    scoresToLog=np.vstack((scores1,scores2))
    print(scores2)

    # slr=lrc.logReg(0,0.1,"balanced")
    # actualDCF,minDCF,scores=val.k_fold_fusionV2(slr,scoresToLog,L,k,(0.1,1,1))
    # print("actualDCF:{},minDCF:{}".format(actualDCF,minDCF))

