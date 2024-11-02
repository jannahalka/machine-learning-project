import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
 This helper Standardize data
 -> this achieves: "You are recommended to consider applying some kind of feature scaling to the pixel values as part of your analyses of the data"
 -> used for PCA
    -> especially useful when using PCA for KNN or SVC
'''

def load_and_preprocess_data(file_path):
    # Load training dataset
    fashion_train = np.load(file_path)

    # Convert to DataFrame
    df_train = pd.DataFrame(fashion_train)

    # Separating out the features (first 784 columns are pixel values) and the labels (last column)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Standardizing the data (PCA is sensitive to scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, y_train
