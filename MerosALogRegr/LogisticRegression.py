import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000, tolerance=1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, batch_size=256):
        X = np.array(X)
        y = np.array(y).flatten()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                predictions = sigmoid(linear_pred)

                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
                db = (1 / len(X_batch)) * np.sum(predictions - y_batch)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            if np.linalg.norm(dw) < self.tolerance and abs(db) < self.tolerance:
                break
    
    def predict(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return np.array([0 if y <= 0.5 else 1 for y in y_pred])

    def accuracy(self, y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)
