from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

fashion_train = np.load("./data/fashion_train.npy")
fashion_test = np.load("./data/fashion_test.npy")


X_train_raw = fashion_train[:, :-1]
y_train_raw = fashion_train[:, -1]
X_test_raw = fashion_test[:, :-1]
y_test_raw = fashion_test[:, -1]

tree = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [10],
    "min_samples_split": [5],
    "min_samples_leaf": [15],
}

grid_search = GridSearchCV(tree, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_raw, y_train_raw)

print(f"Best parameters for Decision Tree: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

tree_best = grid_search.best_estimator_

y_test_pred = tree_best.predict(X_test_raw)
test_accuracy = accuracy_score(y_test_raw, y_test_pred)

print(f"Test accuracy for Decision Tree on raw data: {test_accuracy:.4f}")
print(f"Test error for Decision Tree on raw data: {1 - test_accuracy:.4f}")

