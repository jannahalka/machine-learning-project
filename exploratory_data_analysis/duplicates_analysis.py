import numpy as np
import pandas as pd

# Load your dataset from the .npy file
fashion_train = np.load('../data/fashion_training.npy')

# Convert to a Pandas DataFrame
df_train = pd.DataFrame(fashion_train)

# Check for duplicate rows
duplicate_rows = df_train.duplicated()

# Print whether any duplicates were found
if duplicate_rows.any():
    print(f"Number of duplicate rows found: {duplicate_rows.sum()}")
    print("Duplicate rows:")
    print(df_train[duplicate_rows])
else:
    print("No duplicate rows found.")


'''
    No duplicate rows found for training and test dataset
'''