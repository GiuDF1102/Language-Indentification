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

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.1, balanced=True)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTrainQuadratic)

    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print(f"Accuracy: {val.calc_accuracy(labels, labels_pred)}")

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.1, balanced=False)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTrainQuadratic)

    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print(f"Accuracy: {val.calc_accuracy(labels, labels_pred)}")

    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")
    
        