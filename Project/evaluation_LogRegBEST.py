import validation as val
import data_utils as du
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import math_utils as mu
import numpy as np
import data_visualization as dv

#LOADING DATASET
labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

#FEATURES EXPANSION
featuresTrainQuadratic = du.features_expansion(features)
featuresTrainQuadraticZNorm = mu.z_score(featuresTrainQuadratic)
featuresZNorm = mu.z_score(features)
lambdaBest=127


with open("QLOGREG_PI_PITILDE.txt","w") as f:
    for piT in [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20]:
        CprimList=[]
        for piTilde in [0.1,0.5]:
            logRegObj = lrc.logReg(lambdaBest, piT, "balanced")
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (piTilde, 1, 1))
            print("Quadratic LogReg Balanced Not Normalized, minDCF with piT {} and piTilde {} no PCA and  is {}".format(piT, piTilde, minDCF),file=f)
            CprimList.append(minDCF)
        Cprim=np.array(CprimList).mean(axis=0)
        print("Quadratic LogReg Balanced Not Normalized, Cprim with piT {}  no PCA and  is {}".format(piT,Cprim ),file=f)
