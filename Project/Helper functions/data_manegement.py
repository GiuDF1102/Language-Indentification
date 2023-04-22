import numpy as np

def load(file_name, map):
    file_path = file_name

    label_list = []
    features_list = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                splitted_line = line.removesuffix("\n").split(",")
                label = map[splitted_line.pop().lower()]
                label_list.append(label)
                features_list.append([float(elem) for elem in splitted_line])
            except:
                pass                            

    return (np.array(label_list, dtype=int), np.array(features_list, dtype=float).T)