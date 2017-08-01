import os
import pickle

import numpy as np

projectroot = os.path.expanduser("~/SciProjects/Project_Spam/")

floatX = "float64"


def pull_file(fn, samples):
    edim = 300
    ydim = 50
    # samples = 3987

    mat = np.zeros((samples, ydim, edim), dtype=floatX)
    Y = np.zeros((samples,), dtype=bool)

    s = 0
    y = 0
    for line in open(projectroot + fn):
        if line[0:6] == "label=":
            s += 1
            Y[s - 1] = bool(int(line[6]))
            y = 0
        else:
            mat[s-1, y] = np.array(
                [x for x in line.strip().split(" ", edim)]
            ).astype(floatX)
            y += 1
    return mat, Y[:, None]


def pull_data():
    try:
        alldata = pickle.load(open(projectroot + "alldata.pck", "rb"))
    except Exception as E:
        print("load error:", E, "\nRebuilding data pickle...")
        X, Y = pull_file("spamdata.txtB", 3356)
        Xt, Yt = pull_file("spamdata.txtA", 630)
        alldata = (X, Y, Xt, Yt)
        pickle.dump(alldata, open(projectroot + "alldata.pck", "wb"))
    return alldata
