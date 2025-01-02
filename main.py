import numpy as np
from classifiers.decision_tree.own_implementation import DecisionTreeClassifier
from classifiers.nn.own import FashionNeuralNetwork


if __name__ == "__main__":
    # Instantiate and train
    nn = FashionNeuralNetwork(
        hidden_units=128,
        seed=42
    )
    nn.train(epochs=1000, lr=0.1, print_every=100)

    # Evaluate on test set
    test_acc = nn.accuracy(nn.X_test, nn.y_test) * 100
    print(f"Test accuracy: {test_acc:.2f}%")
