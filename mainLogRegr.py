import numpy as np
from MatrixProcessing import MatrixProcessing
from AdaBoost import AdaBoost  # <-- Import AdaBoost instead of LogisticRegression
import os
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting the model")

# Initialize processor
processor = MatrixProcessing(top_words=1000, skip_top_words=50, skip_least_frequent=20)

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
adaboost = AdaBoost(n_clf=20)  # n_clf= number of weak learners (decision stumps)
adaboost.fit(binary_vectors_train, y_train_transformed)

# Predict on train and test
y_pred_train_ada = adaboost.predict(binary_vectors_train)
y_pred_test_ada = adaboost.predict(binary_vectors_test)

# Convert predictions back to [0, 1]
y_pred_train_ada = np.where(y_pred_train_ada == -1, 0, 1)
y_pred_test_ada  = np.where(y_pred_test_ada == -1, 0, 1)

# -- Compute metrics --
def calculate_metrics(y_true, y_pred):
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

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

# Evaluate
train_accuracy = accuracy(y_pred_train_ada, y_train)
test_accuracy  = accuracy(y_pred_test_ada, y_test)

precision_train, recall_train, f1_train = calculate_metrics(y_train, y_pred_train_ada)
precision_test, recall_test, f1_test   = calculate_metrics(y_test, y_pred_test_ada)

print("\n=== AdaBoost Results ===")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Precision (class=1): {precision_test:.4f}")
print(f"Recall    (class=1): {recall_test:.4f}")
print(f"F1-Score  (class=1): {f1_test:.4f}")

# If you want a confusion matrix or a metrics matrix for both classes:
def calculate_metrics_matrix(y_true, y_pred):
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

metrics_matrix = calculate_metrics_matrix(y_test, y_pred_test_ada)
print("\nModel Evaluation Metrics for both categories:")
print("+" + "-" * 50 + "+")
for row in metrics_matrix:
    print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
print("+" + "-" * 50 + "+")

# Example of how you might want to visualize something from AdaBoost
# (You won't have iteration metrics as we do in LogisticRegression,
#  but you could store intermediate errors or alpha values if you choose.)

print("\nDone!")
