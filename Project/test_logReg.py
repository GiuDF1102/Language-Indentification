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
    orig_stdout = sys.stdout
    f = open('out_log_reg.txt', 'w')
    sys.stdout = f

    start_time = datetime.now()

    #LOAD DATA
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

    #FEATURES EXPANSION
    featuresTrainQuadratic = du.features_expansion(features)

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.5, balanced=True)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTrainQuadratic)

    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print(f"Accuracy: {val.accuracy(labels, labels_pred)}")

    #LAMBDA 
    logRegObj = lrc.logReg(1, pi=0.5, balanced=False)
    logRegObj.train(featuresTrainQuadratic, labels)
    labels_pred = logRegObj.transform(featuresTrainQuadratic)

    #PRINT ACCURACY
    print("--------- ACCURACY ----------")
    print(f"Accuracy: {val.accuracy(labels, labels_pred)}")

    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")
    
    sys.stdout = orig_stdout
    f.close()
        