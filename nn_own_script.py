from classifiers.nn.own import FashionNeuralNetwork

# Instantiate and train
nn = FashionNeuralNetwork(hidden_units=128, seed=42)
nn.train(epochs=1000, lr=0.1, print_every=100)

# Evaluate on test set
test_acc = nn.accuracy(nn.X_test, nn.y_test) * 100
print(f"Test accuracy: {test_acc:.2f}%")
