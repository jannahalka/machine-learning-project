import numpy as np
from sklearn.metrics import accuracy_score
from classifiers.decision_tree.own_implementation import DecisionTreeClassifier

# Load the data
fashion_train = np.load("data/fashion_train.npy")
fashion_test = np.load("data/fashion_test.npy")

# Separate features (X) and labels (y)
X_train, y_train = fashion_train[:, :-1], fashion_train[:, -1]
X_test, y_test = fashion_test[:, :-1], fashion_test[:, -1]

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=10)  # You can adjust max_depth as needed
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

