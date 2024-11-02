from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Load the PCA-transformed training and test data
fashion_train = np.load('../../../data/fashion_train.npy')
fashion_test = np.load('../../../data/fashion_test.npy')


# Separate features and labels
X_train_raw = fashion_train[:, :-1]  # First 784 columns are pixel values
y_train_raw = fashion_train[:, -1]   # Last column is the label (class)
X_test_raw = fashion_test[:, :-1]
y_test_raw = fashion_test[:, -1]

# Standardize the raw data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Initialize a Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)

# Set up a parameter grid for cross-validation to tune the model
# param_grid = {
#     'max_depth': [2, 3, 5, 10, 20, 30, None],
#     'min_samples_split': [2, 5, 10, 15, 20],
#     'min_samples_leaf': [1, 2, 4, 5, 10, 15,20, 50, 100],
#     'criterion': ["gini", "entropy"]
# }

param_grid = {
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [15],
}

# Use GridSearchCV to find the best hyperparameters (gini impurity default)
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_raw)

# Print the best parameters
print(f"Best parameters for Decision Tree: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Train the final decision tree with the best parameters
tree_best = grid_search.best_estimator_

# Predict and evaluate on the test set
y_test_pred = tree_best.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test_raw, y_test_pred)

# Print the test accuracy and test error
print(f"Test accuracy for Decision Tree on raw data: {test_accuracy:.4f}")
print(f"Test error for Decision Tree on raw data: {1 - test_accuracy:.4f}")



