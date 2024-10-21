'''
    -> refactor (duplicity in code and here use PCA(n_components=0.90, whiten=True) from PCA.py
    -> look at z-score and maybe adjust threshold
    -> at the end discuss if we want to get rid of outliers
'''


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load training dataset
fashion_train = np.load('../data/fashion_train.npy')

# Convert to DataFrame
df_train = pd.DataFrame(fashion_train)

# Separating out the features (first 784 columns are pixel values) and the labels (last column)
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]  # The labels (categories)

# Standardizing the data (PCA is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply PCA to reduce dimensions (keep enough components to explain 95% variance)
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_scaled)

# Compute Z-scores for the PCA-transformed data
z_scores_pca = np.abs(stats.zscore(X_train_pca))

# Set threshold for detecting outliers in PCA space (commonly 3, adjust as needed)
threshold = 3

# Identify outliers in PCA-transformed data
outliers_pca = np.where(z_scores_pca > threshold)

# Print information about outliers
print(f"Outliers found in {len(np.unique(outliers_pca[0]))} rows after PCA.")

# Optionally, remove rows with outliers
# X_train_pca_clean = X_train_pca[(z_scores_pca < threshold).all(axis=1)]
# y_train_clean = y_train[(z_scores_pca < threshold).all(axis=1)]
#
# print(f"Dataset shape after outlier removal in PCA space: {X_train_pca_clean.shape}")
