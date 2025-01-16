import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples, dtype=int)
        feature_column = X[:, self.feature_idx]

        # If polarity=1, we predict -1 when feature < threshold
        # If polarity=-1, we predict -1 when feature >= threshold, etc.
        if self.polarity == 1:
            predictions[feature_column < self.threshold] = -1
        else:
            predictions[feature_column > self.threshold] = -1

        return predictions

class AdaBoost:
    def __init__(self, n_clf=1000):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: array of labels in {-1, +1}
        """
        n_samples, n_features = X.shape
        print("n_samples: ", n_samples)
        print("n_features: ", n_features)
        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        for t in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            
            if self.n_clf%100==0:
                print(f"Iteration: {self.n_clf}")

            # For each feature, try all possible thresholds
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                threshold = (thresholds[0] + thresholds[-1]) / 2  # Compute midpoints
                polarity = 1
                predictions = np.ones(n_samples, dtype=int)
                predictions[X_column < threshold] = -1

                # Calculate weighted error
                error = np.sum(w[y != predictions])

                # If error > 0.5, swap polarity
                if error > 0.5:
                    error = 1 - error
                    polarity = -1

                # Keep the best (lowest) error stump
                if error < min_error:
                    clf.polarity = polarity
                    clf.threshold = threshold
                    clf.feature_idx = feature_i
                    min_error = error

            EPS = 1e-10
            # Compute alpha
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Update weights
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # Save this stump
            self.clfs.append(clf)

    def predict(self, X):
        """
        Aggregates predictions from each stump weighted by alpha.
        Returns an array of -1 or +1.
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)
    # -- Compute metrics --
    def calculate_metrics(self,y_true, y_pred):
        """Calculate precision, recall, F1 for class '1'."""
        TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
        FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
        FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score  = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1_score

    def accuracy(self,y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)
    def calculate_metrics_matrix(self,y_true, y_pred):
        # Class 1 metrics
        TP_1 = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
        FP_1 = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
        FN_1 = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))
        precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) > 0 else 0
        recall_1    = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) > 0 else 0
        f1_1        = (
            2 * (precision_1 * recall_1) / (precision_1 + recall_1)
            if (precision_1 + recall_1) > 0
            else 0
        )

        # Class 0 metrics
        TP_0 = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
        FP_0 = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))
        FN_0 = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
        precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
        recall_0    = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
        f1_0        = (
            2 * (precision_0 * recall_0) / (precision_0 + recall_0)
            if (precision_0 + recall_0) > 0
            else 0
        )

        # Micro-average
        TP = TP_1 + TP_0
        FP = FP_1 + FP_0
        FN = FN_1 + FN_0
        precision_micro = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_micro    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_micro        = (
            2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
            if (precision_micro + recall_micro) > 0
            else 0
        )

        # Macro-average
        precision_macro = (precision_1 + precision_0) / 2
        recall_macro    = (recall_1 + recall_0) / 2
        f1_macro        = (f1_1 + f1_0) / 2

        metrics_matrix = np.array([
            ["Category",      "Precision",       "Recall",          "F1-Score"],
            ["Class 0",       precision_0,       recall_0,          f1_0],
            ["Class 1",       precision_1,       recall_1,          f1_1],
            ["Micro-Average", precision_micro,   recall_micro,      f1_micro],
            ["Macro-Average", precision_macro,   recall_macro,      f1_macro],
        ])
        return metrics_matrix