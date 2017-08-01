import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from util import pull_file


def get_model(model):
    if model.lower() == "svm":
        Model = SVC
    elif model.lower() == "nb":
        Model = GaussianNB
    else:
        raise ValueError("No such model: " + model)
    return Model()


def preprocess_data(validation_split):
    X, Y = pull_file("spamdata.txtA")
    N = len(X)
    arg = np.arange(N)
    np.random.shuffle(arg)
    validN = int(validation_split * N)
    validX, trainX = np.split(X, [validN, N])
    validY, trainY = np.split(Y, [validN, N])
    return trainX, trainY, validX, validY


def main(model="svm"):
    tX, tY, vX, vY = preprocess_data(validation_split=0.1)

    print("Fitting SVM to {} datapoints".format(len(tX)))
    model = get_model(model)
    model.fit(tX, tY)
    print("Evaluating SVM on {} datapoints".format(len(vX)))
    pred = model.predict(vX)
    # noinspection PyUnresolvedReferences
    acc = (pred == vY).mean()
    print("Accuracy: {:.2%}".format(acc))


if __name__ == '__main__':
    if input("Warning! This script is a memory-hog. Continue? [y]/n > ") == "n":
        quit(0)
    main(model="svm")
