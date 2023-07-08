import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_time = datetime.now()

    #LOAD DATA
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

    #FEATURES EXPANSION
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTestQuadratic = du.features_expansion(features_test)

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.1, balanced=True)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTestQuadratic)
    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print((labels_pred == labels_test).sum() / len(labels_test))
    print("CONFUSION MATRIX")
    print(val.confusion_matrix(labels_test, labels_pred).get_confusion_matrix())

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.1, balanced=False)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTestQuadratic)

    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print((labels_pred == labels_test).sum() / len(labels_test))
    print("CONFUSION MATRIX")
    print(val.confusion_matrix(labels_test, labels_pred).get_confusion_matrix())

    # plt.figure()
    
    # #KFOLD LAMBDA
    # lambdas=np.logspace(-5, 5, num=50)
    # k=3
    # minDCFlist = []
    # for l in lambdas:
    #     logRegObj = lrc.logReg(l, pi=0.5, balanced=True)
    #     actualDCF, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, k, (0.5, 1, 1), "Log Reg with lambda {}".format(l))
    #     minDCFlist.append(minDCF)
    #     print("{}, minDCF with lambda {} is {}".format(0.5, l, minDCF))
    # plt.plot(lambdas, minDCFlist, label="minDCF", color="red")

    # #KFOLD LAMBDA
    # lambdas=np.logspace(-5, 5, num=50)
    # k=3
    # minDCFlist = []
    # for l in lambdas:
    #     logRegObj = lrc.logReg(l, pi=0.1, balanced=True)
    #     actualDCF, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, k, (0.1, 1, 1), "Log Reg with lambda {}".format(l))
    #     minDCFlist.append(minDCF)
    #     print("{}, minDCF with lambda {} is {}".format(0.1, l, minDCF))
    # plt.plot(lambdas, minDCFlist, label="minDCF", color="blue")

    # plt.xscale("log")
    # plt.xlabel("λ")
    # plt.ylabel("minDCF")
    # plt.legend(title="Prior", loc="upper right", labels=["0.5", "0.1"])
    
    # plt.savefig("logReg_lambda.svg")    
    
    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")
    
        