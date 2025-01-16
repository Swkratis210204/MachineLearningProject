import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, tolerance=1e-6, alpha=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
    def fit(self, X, y, batch_size=256):
        X = np.array(X)
        y = np.array(y).flatten()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Initialize lists to store metrics
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

        for i in range(self.n_iter):
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                predictions = sigmoid(linear_pred)

                # Gradient calculation
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
                db = (1 / len(X_batch)) * np.sum(predictions - y_batch)

                # Add L1 regularization to gradient
                dw += (self.alpha / n_samples) * np.sign(self.weights)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Predictions on the entire dataset
            y_pred = self.predict(X)

            # Calculate metrics at this iteration
            precision, recall, f1_score = self.calculate_metrics(y, y_pred)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1_scores.append(f1_score)

            # Print metrics every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")

            # Check convergence
            if np.linalg.norm(dw) < self.tolerance and abs(db) < self.tolerance:
                print(f"Convergence reached at iteration {i + 1}")
                break

    
    def predict(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return np.array([0 if y <= 0.5 else 1 for y in y_pred])

    def accuracy(self, y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)
    
    def calculate_metrics(self,y_true, y_pred):
        # Υπολογισμός True Positives, False Positives, False Negatives
        TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
        FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
        FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

        # Precision, Recall, F1-Score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score
    
    def calculate_metrics_matrix(self, y_true, y_pred):
        # Υπολογισμός μετρήσεων για την κατηγορία 1
        TP_1 = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
        FP_1 = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
        FN_1 = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

        precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) > 0 else 0
        recall_1 = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) > 0 else 0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

        # Υπολογισμός μετρήσεων για την κατηγορία 0
        TP_0 = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
        FP_0 = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))
        FN_0 = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))

        precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
        recall_0 = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

        # Υπολογισμός μικρο-μέσου (micro-average)
        TP = TP_1 + TP_0
        FP = FP_1 + FP_0
        FN = FN_1 + FN_0
        precision_micro = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_micro = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

        # Υπολογισμός μακρο-μέσου (macro-average)
        precision_macro = (precision_1 + precision_0) / 2
        recall_macro = (recall_1 + recall_0) / 2
        f1_macro = (f1_1 + f1_0) / 2

        # Δημιουργία πίνακα αποτελεσμάτων
        metrics_matrix = np.array([
            ["Category", "Precision", "Recall", "F1-Score"],
            ["Class 0", precision_0, recall_0, f1_0],
            ["Class 1", precision_1, recall_1, f1_1],
            ["Micro-Average", precision_micro, recall_micro, f1_micro],
            ["Macro-Average", precision_macro, recall_macro, f1_macro],
        ])

        return metrics_matrix
    