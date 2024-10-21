import numpy as np

data_training = np.load("../data/fashion_train.npy")
print(data_training.shape)

data_test = np.load("../data/fashion_test.npy")
print(data_test.shape)