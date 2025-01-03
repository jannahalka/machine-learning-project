import numpy as np
import pandas as pd

# Load your datasets from the 'data' directory
fashion_train = np.load("../data/fashion_train.npy")
fashion_test = np.load("../data/fashion_test.npy")

# Combine both datasets to check the imbalance across the entire dataset
combined_data = np.concatenate((fashion_train, fashion_test), axis=0)

# Convert both datasets into DataFrames for easier manipulation
df_train = pd.DataFrame(fashion_train)
df_test = pd.DataFrame(fashion_test)
df_combined = pd.DataFrame(combined_data)

# Check for missing values in training data
missing_train = df_train.isna().sum()

# Check for missing values in test data
missing_test = df_test.isna().sum()

# Check for missing values in combined data
missing_combined = df_combined.isna().sum()

# Display missing values count for each dataset
print("Missing values in training data:")
print(missing_train[missing_train > 0])  # Only display columns with missing values

print("\nMissing values in test data:")
print(missing_test[missing_test > 0])

print("\nMissing values in combined data:")
print(missing_combined[missing_combined > 0])

"""
    No missing values in the training data, test data, or the combined dataset
    This approach takes a look into each column and then sum up the missing values
"""
