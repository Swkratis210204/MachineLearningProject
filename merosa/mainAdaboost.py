import sys
import numpy as np
from AdaBoost import AdaBoost  # <-- Import AdaBoost instead of LogisticRegression
import os
import matplotlib.pyplot as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MatrixProcessing import MatrixProcessing

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting the model")

# Initialize processor
processor = MatrixProcessing(top_words=10000, skip_top_words=50, skip_least_frequent=20)

# Load and filter data
x_train, y_train, x_test, y_test = processor.load_data()

# Convert reviews to binary vectors
binary_vectors_train = processor.map_reviews_to_binary_vectors(x_train)
binary_vectors_test = processor.map_reviews_to_binary_vectors(x_test)

# Convert binary vectors to NumPy arrays
binary_vectors_train = np.array(binary_vectors_train)
binary_vectors_test = np.array(binary_vectors_test)

# Ensure y_train and y_test are NumPy arrays
y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()

if binary_vectors_train.ndim == 1:
    binary_vectors_train = binary_vectors_train.reshape(1, -1)
if binary_vectors_test.ndim == 1:
    binary_vectors_test = binary_vectors_test.reshape(1, -1)

# Calculate top words based on information gain (IG)
top_word_indices = processor.calculate_information_gain(binary_vectors_train, y_train, top_k=1000)
top_word_indices = top_word_indices[: binary_vectors_train.shape[1]]

binary_vectors_train = binary_vectors_train[:, top_word_indices]
binary_vectors_test = binary_vectors_test[:, top_word_indices]

# --- TRANSFORM LABELS FROM {0,1} TO {-1, +1} ---
y_train_transformed = np.where(y_train == 0, -1, 1)
y_test_transformed = np.where(y_test == 0, -1, 1)

# Train AdaBoost
print("Training AdaBoost...")
adaboost = AdaBoost(n_clf=50)  # n_clf= number of weak learners (decision stumps)
adaboost.fit(binary_vectors_train, y_train_transformed)

# Predict on train and test
y_pred_train_ada = adaboost.predict(binary_vectors_train)
y_pred_test_ada = adaboost.predict(binary_vectors_test)

# Convert predictions back to [0, 1]
y_pred_train_ada = np.where(y_pred_train_ada == -1, 0, 1)
y_pred_test_ada  = np.where(y_pred_test_ada == -1, 0, 1)

# Evaluate
train_accuracy = adaboost.accuracy(y_pred_train_ada, y_train)
test_accuracy  = adaboost.accuracy(y_pred_test_ada, y_test)

precision_train, recall_train, f1_train = adaboost.calculate_metrics(y_train, y_pred_train_ada)
precision_test, recall_test, f1_test   = adaboost.calculate_metrics(y_test, y_pred_test_ada)

print("\n=== AdaBoost Results ===")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Precision (class=1): {precision_test:.4f}")
print(f"Recall    (class=1): {recall_test:.4f}")
print(f"F1-Score  (class=1): {f1_test:.4f}")


metrics_matrix = adaboost.calculate_metrics_matrix(y_test, y_pred_test_ada)
print("\nModel Evaluation Metrics for both categories:")
print("+" + "-" * 50 + "+")
for row in metrics_matrix:
    print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
print("+" + "-" * 50 + "+")

print("\nDone!")