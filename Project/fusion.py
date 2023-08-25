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

def k_fold_fusionV3(learner1, learner2,x1,x2,labels,k, workingPoint, name):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X1, Y = shuffle(x1.T, labels, random_state=0)
    X2, Y = shuffle(x2.T, labels, random_state=0)
    X1_splitted = np.array_split(X1, k)
    X2_splitted = np.array_split(X2, k)
    y_splitted = np.array_split(Y, k)
    concat_scores1 = []
    concat_scores2 = []
    concat_predicted1 = []
    concat_predicted2 = []
    for i in range(k): #for each fold
        X1_folds = X1_splitted.copy()
        X2_folds = X2_splitted.copy()
        y_folds = y_splitted.copy()
        X1_val = X1_folds.pop(i).T
        X2_val = X2_folds.pop(i).T
        y_val = y_folds.pop(i)
        X1_train = np.vstack(X1_folds).T
        X2_train = np.vstack(X2_folds).T
        y_train = np.hstack(y_folds)
        learner1.train(X1_train, y_train)
        learner2.train(X2_train, y_train)
        lab_1 = learner1.transform(X1_val)
        lab_2 = learner2.transform(X2_val)
        scores1 = learner1.get_scores()
        scores2 = learner2.get_scores()
        concat_scores1.append(scores1)
        concat_scores2.append(scores2)
    scores = np.zeros((2, len(labels)))
    scores[0] = np.hstack(concat_scores1)
    scores[1] = np.hstack(concat_scores2)

    print("score:{},score_shape:{}".format(scores,scores.shape))
    # score12 = np.vstack((mu.FromColumnToRow(gotscores1),mu.FromColumnToRow (gotscores2)))

    # FUSION
    slr = lrc.logReg(0,0.1,"balanced")
    slr.train(scores,Y)
    w,b=slr.get_params()
    f_s = np.dot(w.T,scores)+b
    lab_s = np.where(f_s>0,1,0)

    actualDCF_ = val.act_DCF(f_s,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(f_s, pi, Cfn, Cfp, Y, lab_s)
    print("not calibrated {}, {}".format(actualDCF_, minDCF_))

    val.get_error_plot(f_s, Cfn, Cfp, Y, lab_s, "not calibrated {}".format(name))

    f_s = mu.FromColumnToRow(f_s)
    # CALIBRATION
    lrcobj = lrc.logReg(0,pi,"balanced")
    lrcobj.train(f_s,Y)
    alpha, beta = lrcobj.get_params()
    
    calibScores = alpha*scores+beta-np.log(pi/(1-pi))
    calibScores = calibScores[0]
    calibLabels = np.where(calibScores>0,1,0)
    
    actualDCF_ = val.act_DCF(calibScores,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    
    val.get_error_plot(calibScores, Cfn, Cfp, Y, calibLabels, "calibrated {}".format(name))
    print("calibrated {}, {}".format(actualDCF_, minDCF_))
    return actualDCF_, minDCF_, scores

def k_fold_fusionV4(learner1, learner2,x1,x2,labels,k, workingPoint, name):
    pi = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    X1, Y = shuffle(x1.T, labels, random_state=0)
    X2, Y = shuffle(x2.T, labels, random_state=0)
    X1_splitted = np.array_split(X1, k)
    X2_splitted = np.array_split(X2, k)
    y_splitted = np.array_split(Y, k)
    concat_scores1 = []
    concat_scores2 = []
    concat_predicted1 = []
    concat_predicted2 = []
    for i in range(k): #for each fold
        X1_folds = X1_splitted.copy()
        X2_folds = X2_splitted.copy()
        y_folds = y_splitted.copy()
        X1_val = X1_folds.pop(i).T
        X2_val = X2_folds.pop(i).T
        y_val = y_folds.pop(i)
        X1_train = np.vstack(X1_folds).T
        X2_train = np.vstack(X2_folds).T
        y_train = np.hstack(y_folds)
        learner1.train(X1_train, y_train)
        learner2.train(X2_train, y_train)
        lab_1 = learner1.transform(X1_val)
        lab_2 = learner2.transform(X2_val)
        scores1 = learner1.get_scores()
        scores2 = learner2.get_scores()
        concat_scores1.append(scores1)
        concat_scores2.append(scores2)
    scores = np.zeros((2, len(labels)))
    
    scores1 = np.hstack(concat_scores1)
    lrcobj = lrc.logReg(0,pi,"balanced")
    print(scores1.shape)
    print(Y.shape)
    print(scores1)
    print(Y)
    lrcobj.train(scores1,Y)
    alpha1, beta1 = lrcobj.get_params()
    calibScores1 = alpha1*scores1+beta1-np.log(pi/(1-pi))
    calibScores1 = calibScores1[0]
    calibLabels1 = np.where(calibScores1>0,1,0)

    scores2 = np.hstack(concat_scores2)
    lrcobj = lrc.logReg(0,pi,"balanced")
    lrcobj.train(scores2,Y)
    alpha2, beta2 = lrcobj.get_params()    
    calibScores2 = alpha2*scores2+beta2-np.log(pi/(1-pi))
    calibScores2 = calibScores2[0]
    calibLabels2 = np.where(calibScores2>0,1,0)

    calibScores1 = mu.FromRowToColumn(calibScores1)
    calibScores2 = mu.FromRowToColumn(calibScores2)
    scores[0] = np.hstack(calibScores1)
    scores[1] = np.hstack(calibScores2)

    print("score:{},score_shape:{}".format(scores,scores.shape))
    # score12 = np.vstack((mu.FromColumnToRow(gotscores1),mu.FromColumnToRow (gotscores2)))

    # FUSION
    slr = lrc.logReg(0,0.1,"balanced")
    slr.train(scores,Y)
    w,b=slr.get_params()
    f_s = np.dot(w.T,scores)+b
    lab_s = np.where(f_s>0,1,0)

    actualDCF_ = val.act_DCF(f_s,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(f_s, pi, Cfn, Cfp, Y, lab_s)
    print("not calibrated {}, {}".format(actualDCF_, minDCF_))

    val.get_error_plot(f_s, Cfn, Cfp, Y, lab_s, "not calibrated {}".format(name))

    f_s = mu.FromColumnToRow(f_s)
    # CALIBRATION
    lrcobj = lrc.logReg(0,pi,"balanced")
    lrcobj.train(f_s,Y)
    alpha, beta = lrcobj.get_params()
    
    calibScores = alpha*scores+beta-np.log(pi/(1-pi))
    calibScores = calibScores[0]
    calibLabels = np.where(calibScores>0,1,0)
    
    actualDCF_ = val.act_DCF(calibScores,pi, Cfn, Cfp, Y, None)
    minDCF_ = val.min_DCF(calibScores, pi, Cfn, Cfp, Y, calibLabels)
    
    val.get_error_plot(calibScores, Cfn, Cfp, Y, calibLabels, "calibrated {}".format(name))
    print("calibrated {}, {}".format(actualDCF_, minDCF_))
    return actualDCF_, minDCF_, scores


def k_fold_error_plot_Cprim(learner1, learner2,x1,x2,labels,k, workingPoint1,workingPoint2,piT, name,nametext,plot=False,perPrior=False):
    filename="risultati{}piT{}.txt".format(nametext,piT)
    pi1 = workingPoint1[0]
    Cfn1 = workingPoint1[1]
    Cfp1 = workingPoint1[2]
    pi2 = workingPoint2[0]
    Cfn2 = workingPoint2[1]
    Cfp2 = workingPoint2[2]
    X1, Y = shuffle(x1.T, labels, random_state=0)
    X2, Y = shuffle(x2.T, labels, random_state=0)
    X1_splitted = np.array_split(X1, k)
    X2_splitted = np.array_split(X2, k)
    y_splitted = np.array_split(Y, k)
    concat_scores1 = []
    concat_scores2 = []
    concat_predicted1 = []
    concat_predicted2 = []
    for i in range(k): #for each fold
        X1_folds = X1_splitted.copy()
        X2_folds = X2_splitted.copy()
        y_folds = y_splitted.copy()
        X1_val = X1_folds.pop(i).T
        X2_val = X2_folds.pop(i).T
        y_val = y_folds.pop(i)
        X1_train = np.vstack(X1_folds).T
        X2_train = np.vstack(X2_folds).T
        y_train = np.hstack(y_folds)
        learner1.train(X1_train, y_train)
        learner2.train(X2_train, y_train)
        concat_predicted1 = learner1.transform(X1_val)
        concat_predicted2 = learner2.transform(X2_val)
        scores1 = learner1.get_scores()
        scores2 = learner2.get_scores()
        concat_scores1.append(scores1)
        concat_scores2.append(scores2)
    gotscores1 = np.hstack(concat_scores1)
    gotpredicted1 = np.hstack(concat_predicted1)
    gotscores2 = np.hstack(concat_scores2)
    gotpredicted2 = np.hstack(concat_predicted2)

    actualDCF1= val.act_DCF(gotscores1,pi1, Cfn1, Cfp1, Y, None)
    minDCF1 = val.min_DCF(gotscores1, pi1, Cfn1, Cfp1, Y, gotpredicted1)
    actualDCF2= val.act_DCF(gotscores2,pi2, Cfn2, Cfp2, Y, None)
    minDCF2 = val.min_DCF(gotscores2, pi2, Cfn2, Cfp2, Y, gotpredicted2)

    if plot==True:
        val.get_error_plot_Cprim(gotscores1,gotscores2, Cfn1, Cfp1, Y, gotpredicted1,gotpredicted2, "{}piT{}".format(name,piT))

        if perPrior==True:
            val.get_error_plot(gotscores1,Cfn1,Cfn1,Y,gotpredicted1,"{}pi{}".format(name,pi1))
            val.get_error_plot(gotscores2,Cfn2,Cfn2,Y,gotpredicted2,"{}pi{}".format(name,pi2))

    gotscores1 = mu.FromColumnToRow(gotscores1)
    lrcobj = lrc.logReg(0,piT,"balanced")
    lrcobj.train(gotscores1,Y)
    alpha, beta = lrcobj.get_params()

    calibScores1 = alpha*gotscores1+beta-np.log(pi1/(1-pi1))
    calibScores1 = calibScores1[0]

    gotscores2 = mu.FromColumnToRow(gotscores2)
    lrcobj = lrc.logReg(0,piT,"balanced")
    lrcobj.train(gotscores2,Y)
    alpha, beta = lrcobj.get_params()
    
    calibScores2 = alpha*gotscores2+beta-np.log(pi2/(1-pi2))
    calibScores2 = calibScores2[0]

    calActualDCF1= val.act_DCF(calibScores1,pi1, Cfn1, Cfp1, Y, None)
    calMinDCF1 = val.min_DCF(calibScores1, pi1, Cfn1, Cfp1, Y, gotpredicted1)
    calActualDCF2= val.act_DCF(calibScores2,pi2, Cfn2, Cfp2, Y, None)
    calMinDCF2 = val.min_DCF(calibScores2, pi2, Cfn2, Cfp2, Y, gotpredicted2)

    if plot==True:
        val.get_error_plot_Cprim(calibScores1,calibScores2, Cfn1, Cfp1, Y, gotpredicted1,gotpredicted2, "{}piT{}calibrated".format(name,piT))

        if perPrior==True:
            val.get_error_plot(calibScores1,Cfn1,Cfn1,Y,gotpredicted1,"{}calibratedpi{}".format(name,pi1))
            val.get_error_plot(calibScores2,Cfn2,Cfn2,Y,gotpredicted2,"{}calibratedpi{}".format(name,pi2))

    

    avg_min_dcf=((minDCF1+minDCF2)/2)
    avg_act_dcf=((actualDCF1+actualDCF2)/2)
    avg_cal_min_dcf=((calMinDCF1+calMinDCF2)/2)
    avg_cal_act_dcf=((calActualDCF1+calActualDCF2)/2)
    with open(filename,"w") as f:
        print("not calibrated model with pi=0.1, actDCF:{}, minDCF:{}".format(actualDCF1,minDCF1),file=f)
        print("not calibrated model with pi=0.5, actDCF:{}, minDCF:{}".format(actualDCF2,minDCF2),file=f)
        print("not calibrated average minDCF(Cprim):{} and average actDCF:{}".format(avg_min_dcf,avg_act_dcf),file=f)
        print("calibrated model with pi=0.1 and piT:{}, actDCF:{}, minDCF:{}".format(piT,calActualDCF1,calMinDCF1),file=f)
        print("calibrated model with pi=0.5 and piT:{}, actDCF:{}, minDCF:{}".format(piT,calActualDCF2,calMinDCF2),file=f)
        print("calibrated with piT:{},average minDCF(Cprim):{} and average actDCF:{}".format(piT,avg_cal_min_dcf,avg_cal_act_dcf),file=f)
    return 


if __name__=="__main__":
    """
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
    val2.get_error_plot(scores, Cfn, Cfp, true_labels, predicted_labels, name):
    actDCF1 = val.act_DCF(f_s,0.1,1,1,L,None)
    minDCF1 = val.min_DCF(f_s, 0.1, 1, 1, L, lab)

    print("-----------------prior 0.1")
    print(actDCF1, minDCF1)

#-------------------------------------

    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    PCA5=dr.PCA(D,5)
    bestLambda=10
    k=5
    lr=lrc.logReg(bestLambda,0.1,"balanced")
    DPCA = du.features_expansion(D)
    scores1=k_fold_fusion(lr,DPCA,L,k,(0.1,1,1))

    svmc=svm.SVM("RBF",balanced=True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    scores2=k_fold_fusion(svmc,PCA5,L,k,(0.1,1,1))

    scoresToLog=np.vstack((scores1,scores2))

    L = shuffle(L, random_state=0)

    slr = lrc.logReg(0,0.1,"balanced")
    slr.train(scoresToLog,L)
    w,b=slr.get_params()
    f_s = np.dot(w.T,scoresToLog)+b

    lab = np.where(f_s>0,1,0)
    Cfn=1
    Cfp=1

    val.get_error_plot(f_s, Cfn, Cfp, L, lab, "Fusion SVM-LR-0.1")
    """
    L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
    
    data1 = du.features_expansion(D)  
    data2 = dr.PCA(D,5)

    #class1=lrc.logReg(10,0.1,"balanced")
    #class2=lrc.logReg(10,0.1,"balanced")
    class1=svm.SVM("RBF",balanced=True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    class2=svm.SVM("RBF",balanced=True,K=0.01,C=0.1,gamma=0.01,piT=0.2)
    
    #class1 = gmm.GMM(2,32,"MVG", "tied")
    #class2 = gmm.GMM(2,32,"MVG", "tied")
    k_fold_error_plot_Cprim(class1,class2,data2,data2,L,5,(0.1,1,1),(0.5,1,1),0.1,"errorPlotCprimRBF","RBF",plot=False,perPrior=False)
    """k_fold_fusionV3(class1,class2,data1,data2,L,5, (0.1,1,1), "Test-QLOG-SVM")"""




