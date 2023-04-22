import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil

def load(file_name):
    file_path = file_name

    label_list = []
    features_list = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                splitted_line = line.removesuffix("\n").split(",")
                label_list.append(splitted_line.pop().lower())
                features_list.append([float(elem) for elem in splitted_line])
            except:
                pass                            

    return (np.array(label_list, dtype=int), np.array(features_list, dtype=float).T)

def get_hist(data, labels, map_classes, map_features):

    shutil.rmtree("histograms")
    os.makedirs("histograms")

    length = len(np.unique(labels))
    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}

    for i in range(length):
        classes_list.append(data[:, (labels == i)])

    for i in range(length+1):
        plt.figure()
        plt.xlabel(inv_map_feats[i])
        for j in range(length):
            plt.hist(classes_list[j][i], bins = 10, density = True, alpha = 0.4, label = inv_map_class[j])
            plt.legend()
            plt.tight_layout()

        plt.savefig("histograms/Histogram {}.svg".format(inv_map_feats[i]))

def get_scatter(data, labels, map_classes, map_features):
    
    shutil.rmtree("scatter_plots")
    os.makedirs("scatter_plots")

    length_feat = len(map_features)
    length_class = len(map_classes)

    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}

    for i in range(length_feat):
        classes_list.append(data[:, (labels == i)])


    for i in range(length_feat):
        for j in range(length_feat):
            plt.figure()
            for k in range(length_class):
                if i != j:
                    plt.xlabel(inv_map_feats[i])
                    plt.ylabel(inv_map_feats[j])
                    plt.scatter(classes_list[k][i], classes_list[k][j], label = inv_map_class[k])
            if i != j:
                plt.legend()
                plt.tight_layout()
                plt.savefig("scatter_plots/Scatter Plot {} x {}.svg".format(inv_map_feats[i], inv_map_feats[j]))

if __name__=="__main__":
    labels, features = load("Test.txt")
    print(features)
    print(labels)