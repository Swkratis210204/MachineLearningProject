import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        """
        Predict using the trained decision stump.
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples, dtype=int)
        feature_column = X[:, self.feature_idx]

        # Polarity determines direction of threshold comparison
        if self.polarity == 1:
            predictions[feature_column < self.threshold] = -1
        else:
            predictions[feature_column > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []  # List of weak learners
        self.alphas = []  # List of weights for the weak learners

    def fit(self, X, y):
        """
        Fit the AdaBoost classifier to the data.

        X: shape (n_samples, n_features), feature matrix.
        y: shape (n_samples,), target labels in {-1, +1}.
        """
        n_samples, n_features = X.shape

        # Initialize weights equally
        w = np.full(n_samples, (1 / n_samples))

        for m in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Iterate over all features and thresholds to find the best decision stump
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                # Try all unique thresholds for the feature
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        # Predict using the current threshold and polarity
                        predictions = np.ones(n_samples, dtype=int)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1

                        # Calculate weighted error
                        error = np.sum(w[y != predictions])

                        # Keep track of the best decision stump
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            # If error is >= 0.5, discard the weak learner and stop as per the pseudocode
            if min_error >= 0.5:
                print(f"Weak learner {m+1} discarded due to high error ({min_error}). Stopping.")
                break

            # Calculate alpha (weight of this weak learner)
            EPS = 1e-10  # To avoid division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Update weights
            predictions = clf.predict(X)
            for i in range(n_samples):
                if y[i] == predictions[i]:
                    w[i] *= np.exp(-clf.alpha)
                else:
                    w[i] *= np.exp(clf.alpha)

            # Normalize weights
            w /= np.sum(w)

            # Save this weak learner
            self.clfs.append(clf)
            self.alphas.append(clf.alpha)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        n_samples = X.shape[0]
        clf_preds = np.zeros(n_samples)

        # Weighted majority vote
        for clf, alpha in zip(self.clfs, self.alphas):
            clf_preds += alpha * clf.predict(X)

        return np.sign(clf_preds)
