import numpy as np


class FashionNeuralNetwork:
    def __init__(
        self,
        train_path="./data/fashion_train.npy",
        test_path="./data/fashion_test.npy",
        hidden_units=128,
        seed=42,
    ):
        np.random.seed(seed)

        training = np.load(train_path)
        self.X_train = training[:, :-1]
        self.y_train = training[:, -1]

        test = np.load(test_path)
        self.X_test = test[:, :-1]
        self.y_test = test[:, -1]

        self.X_train = self.X_train.astype(np.float32) / 255.0
        self.X_test = self.X_test.astype(np.float32) / 255.0

        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        self.num_features = self.X_train.shape[1]
        self.hidden_units = hidden_units
        self.num_classes = 10

        self.w1 = 0.01 * np.random.randn(self.num_features, hidden_units)
        self.b1 = np.zeros((hidden_units,))

        # w2: (hidden_units, 10), b2: (10,)
        self.w2 = 0.01 * np.random.randn(hidden_units, self.num_classes)
        self.b2 = np.zeros((self.num_classes,))

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Applies the ReLU function elementwise."""
        return np.maximum(0, x)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Applies softmax row-wise.
        x shape: (N, C) -> returns shape: (N, C)
        """
        x_shifted = x - np.max(x, axis=1, keepdims=True)  # for numerical stability
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        """
        Forward pass:
          1) z1 = X.dot(w1) + b1
          2) a1 = ReLU(z1)
          3) logits = a1.dot(w2) + b2
        Returns: (a1, logits)
        """
        z1 = X.dot(self.w1) + self.b1
        a1 = self.relu(z1)
        logits = a1.dot(self.w2) + self.b2
        return a1, logits

    def cross_entropy(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes mean cross-entropy loss for multi-class classification.
        logits: (N, 10)
        y_true: (N,) with class indices [0..9]
        """
        probs = self.softmax(logits)  # (N, 10)
        N = y_true.shape[0]
        eps = 1e-9
        correct_probs = probs[
            np.arange(N), y_true
        ]  # pick the prob of the correct class
        loss = -np.mean(np.log(correct_probs + eps))
        return loss

    def backpropagation(
        self, X: np.ndarray, a1: np.ndarray, logits: np.ndarray, y_true: np.ndarray
    ):
        """
        Computes gradients via backprop:
          dW1, db1, dW2, db2
        """
        N = X.shape[0]

        # 1. Softmax
        probs = self.softmax(logits)

        # 2. Convert y to one-hot
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), y_true] = 1.0

        # 3. Gradient wrt logits
        dlogits = (probs - y_onehot) / N

        # 4. Grad for w2, b2
        dW2 = a1.T.dot(dlogits)
        db2 = np.sum(dlogits, axis=0)

         # 5. Backprop to hidden layer
        dA1 = dlogits.dot(self.w2.T)

        # 6. Apply ReLU derivative
        dZ1 = dA1 * (a1 > 0)

        # 7. Grad for w1, b1
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2, lr):
        """
        Update parameters using gradient descent.
        """
        self.w1 -= lr * dW1
        self.b1 -= lr * db1
        self.w2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, epochs=10, lr=0.1, print_every=1):
        """
        Simple training loop with batch gradient descent over the entire training set.
        """
        loss_history = []
        for epoch in range(1, epochs + 1):
            # Forward pass
            a1, logits = self.forward(self.X_train)

            # Compute loss
            loss = self.cross_entropy(logits, self.y_train)
            loss_history.append(loss)

            # Backprop
            dW1, db1, dW2, db2 = self.backpropagation(
                self.X_train, a1, logits, self.y_train
            )

            # Update params
            self.update_parameters(dW1, db1, dW2, db2, lr)

            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss:.4f}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        _, logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        preds = self.predict(X)
        return np.mean(preds == y_true)
