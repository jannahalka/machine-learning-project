import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load your datasets from the 'data' directory
fashion_train = np.load("../data/fashion_train.npy")
fashion_test = np.load("../data/fashion_test.npy")

# Convert both datasets into DataFrames for easier manipulation
df_train = pd.DataFrame(fashion_train)
df_test = pd.DataFrame(fashion_test)

# Extract the category column (column 785) from both training and test data
category_counts_train = df_train.iloc[:, 784].value_counts()
category_counts_test = df_test.iloc[:, 784].value_counts()

# Combine both datasets to check the imbalance across the entire dataset
combined_data = np.concatenate((fashion_train, fashion_test), axis=0)
df_combined = pd.DataFrame(combined_data)
category_counts_combined = df_combined.iloc[:, 784].value_counts()


# Function to plot the distribution of categories
def plot_category_distribution(category_counts, title):
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.show()


# Plot for training data
plot_category_distribution(
    category_counts_train, "Category Distribution (Training Data)"
)

# Plot for test data
plot_category_distribution(category_counts_test, "Category Distribution (Test Data)")

# Plot for combined data
plot_category_distribution(
    category_counts_combined, "Category Distribution (Combined Data)"
)


"""
    Dataset is balanced
"""
