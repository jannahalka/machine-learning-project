import numpy as np
from classifiers.classifier import DecisionTreeClassifier

training = np.load("./data/fashion_train.npy")
X = training[:, :-1]  # all columns but the last
y = training[:, -1]  # expected to be from 0 to n_classes - 1


# Fit data.
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
