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
    def __init__(self, n_clf=5):
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
