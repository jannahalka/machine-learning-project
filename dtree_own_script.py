import numpy as np
from sklearn.metrics import accuracy_score
from classifiers.decision_tree.own_implementation import DecisionTreeClassifier

fashion_train = np.load("data/fashion_train.npy")
fashion_test = np.load("data/fashion_test.npy")

X_train, y_train = fashion_train[:, :-1], fashion_train[:, -1]
X_test, y_test = fashion_test[:, :-1], fashion_test[:, -1]

clf = DecisionTreeClassifier(max_depth=10) # Set to 10 since we got the best result with 10 when using sklearn
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

