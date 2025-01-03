"""
    -> toto spoznamkovat lebo pri vytvarani vsetkych classifierov treba pouzivat tento PCA a nie "raw data"
    -> konkretne budeme pouzivat pca = PCA(n_components=0.90, whiten=True) (done)
        -> konkretne:
            -> "../data/train_data_for_classifiers" (done)
            -> "../data/test_data_for_classifiers" (done)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from helpers.standardize_data import load_and_preprocess_data

"""
    ->PCA for classifiers
"""

# Load and preprocess the data
X_train_scaled, y_train = load_and_preprocess_data("../data/fashion_train.npy")

# Step 2: Applying PCA to find all components
#     -> white=True for KNN and Neural Network
pca = PCA(n_components=0.90, whiten=True)
X_pca = pca.fit_transform(X_train_scaled)

# Create the models directory if it doesn't exist
joblib.dump(pca, "../models/pca_model.pkl")

# Step 3: Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Step 4: Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance)

# Step 5: Find the number of components explaining at least 90% of the variance
n_components_90_variance = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(
    f"Number of principal components that explain 90% of variance: {n_components_90_variance}"
)

# Step 6: Calculate variance explained by the first 2 principal components
variance_first_2_pcs = np.sum(explained_variance[:2])
print(
    f"Variance explained by the first 2 principal components: {variance_first_2_pcs * 100:.2f}%"
)

# Step 7: Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(cumulative_explained_variance)
plt.axhline(y=0.90, color="r", linestyle="--", label="90% Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.legend()
plt.grid(True)
plt.show()


"""
    -> PCA for visualization
"""
# Step 8: Applying PCA to reduce data to 2 components (for visualization)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train_scaled)

# Step 9: Plotting the data points in 2D space using PC1 and PC2
plt.figure(figsize=(10, 8))

# Create a scatter plot with 5 distinct colors for 5 categories
categories = [0, 1, 2, 3, 4]
colors = ["red", "blue", "green", "orange", "purple"]

for category, color in zip(categories, colors):
    subset = X_pca_2d[y_train == category]
    plt.scatter(
        subset[:, 0],
        subset[:, 1],
        label=f"Category {category}",
        facecolors="none",
        edgecolors=color,
        alpha=0.7,
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Projection of Data onto Principal Components 1 and 2 (36.32% variance)")

# Adding a legend to represent each category
plt.legend(title="Categories")
plt.grid(True)
plt.show()

# Step 10: Get and print the loadings (components) for PC1 and PC2
loadings = pca_2d.components_
loadings_df = pd.DataFrame(loadings.T, columns=["PC1", "PC2"])

print("Loadings for PC1 and PC2:")
print(loadings_df)


"""
 -> save test data with PCA
"""
# Load the pre-trained PCA model
pca = joblib.load("../models/pca_model.pkl")

# Load and preprocess the test data (scaling is required before applying PCA)
X_test_scaled, y_test = load_and_preprocess_data("../data/fashion_test.npy")

# Apply the same PCA transformation to the test data
X_test_pca = pca.transform(X_test_scaled)

# Optionally, save the PCA-transformed test data for future use
np.save("../data/test_data_for_classifiers/X_test_pca.npy", X_test_pca)
np.save("../data/test_data_for_classifiers/y_test.npy", y_test)

print("PCA applied to test data and saved.")
