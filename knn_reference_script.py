"""
    -> "Baseline classifier"
        -> often does not work well with high dimensional feature space
            -> but PCA reduced from 784 features to 112 PC's
            -> and we also set whitening = True

    -> here maybe explore options with Bayes classifier as theoretical benchmark (not sure if it can be used here)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score

X_train_pca_clean = np.load(
    "./data/train_data_for_classifiers/X_train_pca_clean.npy"
)
y_train_clean = np.load("./data/train_data_for_classifiers/y_train_clean.npy")

X_test_pca = np.load("./data/test_data_for_classifiers/X_test_pca.npy")
y_test = np.load("./data/test_data_for_classifiers/y_test.npy")

knn = KNeighborsClassifier()

param_grid = {"n_neighbors": np.arange(1, 31), "weights": ["uniform", "distance"]}


"""
    cv (number of folds) = 3, 5 and 10 give similar result, 5 is a good option
    -> GridSearchCV is great class for C-V, mention how it works in documentation
"""
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_pca_clean, y_train_clean)

print(f"Best parameters for KNN: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")



best_k = grid_search.best_params_["n_neighbors"]
best_weights = grid_search.best_params_["weights"]

knn_final = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
knn_final.fit(X_train_pca_clean, y_train_clean)

y_test_pred = knn_final.predict(X_test_pca)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test error: {1 - test_accuracy:.4f}")
