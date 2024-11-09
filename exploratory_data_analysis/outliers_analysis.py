'''
    -> refactor (duplicity in code and here use PCA(n_components=0.90, whiten=True) from PCA.py (done)
    -> look at z-score and maybe adjust threshold (done)
    -> at the end discuss if we want to get rid of outliers, in training data (done)
        -> we set threshold on 4.5, which gave us 5% of points as outliers
'''


import numpy as np
import joblib
from scipy import stats
from exploratory_data_analysis.helpers.standardize_data import load_and_preprocess_data

# Load and preprocess the data
X_train_scaled, y_train = load_and_preprocess_data('../data/fashion_train.npy')

# Apply PCA to reduce dimensions (keep enough components to explain 95% variance)
pca = joblib.load('../models/pca_model.pkl')
# Apply the pre-trained PCA to the scaled data
X_train_pca = pca.transform(X_train_scaled)

# Compute Z-scores for the PCA-transformed data
z_scores_pca = np.abs(stats.zscore(X_train_pca))

# Set threshold for detecting outliers in PCA space (commonly 3, adjust as needed)
threshold = 4.5

# Identify outliers in PCA-transformed data
outliers_pca = np.where(z_scores_pca > threshold)

# Print information about outliers ->
print(f"Outliers found in {len(np.unique(outliers_pca[0]))} rows after PCA.")

# -> Outliers found in 503 rows after PCA -> 5.03% of training points identified as outliers


# Check how many outliers exist per class
outlier_classes = y_train[np.unique(outliers_pca[0])]  # Extract the classes of the outliers
print(np.bincount(outlier_classes.astype(int)))
# [152  25  96  32 198] in training set, and also overall, most observations are from class 0 and 4 so this should not cause problems


'''
 -> save training data with PCA and without outliers
'''

# remove rows with outliers
X_train_pca_clean = X_train_pca[(z_scores_pca < threshold).all(axis=1)]
y_train_clean = y_train[(z_scores_pca < threshold).all(axis=1)]

# Save the cleaned PCA-transformed dataset (without outliers)
np.save('../data/train_data_for_classifiers/X_train_pca_clean.npy', X_train_pca_clean)

# Save the corresponding cleaned labels (without outliers)
np.save('../data/train_data_for_classifiers/y_train_clean.npy', y_train_clean)

print("Cleaned dataset and labels saved to .npy files.")

