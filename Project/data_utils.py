import numpy as np

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