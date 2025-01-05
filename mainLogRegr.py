import numpy as np
from MatrixProcessing import MatrixProcessing
from LogisticRegression import LogisticRegression
import os
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting the model")

# Initialize processor
processor = MatrixProcessing(top_words=10000, skip_top_words=20, skip_least_frequent=10)

# Load and filter data
x_train, y_train, x_test, y_test = processor.load_data()

#each x,y vector has this form:
"""x:
[2, 2, 20, 47, 111, 439, 3445, 2, 2, 2, 166, 2, 216, 125, 40, 2, 364, 352, 707, 1187, 39, 294, 2, 22, 396, 2, 28, 2, 202, 2, 1109, 23, 94, 2, 151, 111, 211, 469, 2, 20, 2, 258, 546, 1104, 
7273, 2, 2, 38, 78, 33, 211, 2, 2, 2, 2849, 63, 93, 2, 2, 253, 106, 2, 2, 48, 335, 267, 2, 2, 364, 1242, 1179, 20, 2, 2, 1009, 2, 1987, 189, 2, 2, 8419, 2, 2723, 2, 95, 1719, 2, 6035, 2, 3912, 7144, 49, 369, 120, 2, 28, 49, 253, 2, 2, 2, 1041, 2, 85, 795, 2, 2, 481, 2, 55, 78, 807, 2, 375, 2, 1167, 2, 794, 76, 2, 2, 58, 2, 2, 816, 2, 243, 2, 43, 50]
y: 
0
"""
# Convert reviews to binary vectors
binary_vectors_train = processor.map_reviews_to_binary_vectors(x_train)
binary_vectors_test = processor.map_reviews_to_binary_vectors(x_test)

# Convert binary vectors to NumPy arrays
binary_vectors_train = np.array(binary_vectors_train)
binary_vectors_test = np.array(binary_vectors_test)

# Ensure y_train and y_test are NumPy arrays and 1D
y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()

if binary_vectors_train.ndim == 1:
    binary_vectors_train = binary_vectors_train.reshape(1, -1)

if binary_vectors_test.ndim == 1:
    binary_vectors_test = binary_vectors_test.reshape(1, -1)

# Calculate top words based on information gain
top_words = processor.calculate_information_gain(binary_vectors_train, y_train, top_k=1000)

top_word_indices = list(range(1000))  # Map directly to reduced feature space
binary_vectors_train = binary_vectors_train[:, top_word_indices]
binary_vectors_test = binary_vectors_test[:, top_word_indices]

# Train and evaluate the logistic regression model
logRegr = LogisticRegression(tolerance=1e-6)
print(f"Finished processing the reviews, now going into the logistic regression computation")
logRegr.fit(binary_vectors_train, y_train, batch_size=256)

y_pred_train=logRegr.predict(binary_vectors_train)
y_pred_test = logRegr.predict(binary_vectors_test)

# Calculate accuracy
accuracy = logRegr.accuracy(y_pred_test, y_test)

# Calculate precision,recall and f1 score for category 1
precision, recall, f1_score = logRegr.calculate_metrics(y_train, y_pred_train)
print(f"\nLogistic Regression Accuracy: {accuracy}")
print(f"Precision for category 1: {precision}")
print(f"Recall for category 1: {recall}")
print(f"F1-Score for category 1: {f1_score}")
print("Plotting the learning curves now")

metrics_matrix = logRegr.calculate_metrics_matrix(y_test, y_pred_test)
# Print metrics in table format
print("\nModel Evaluation Metrics for both categories:")
print("+" + "-" * 50 + "+")
for row in metrics_matrix:
    print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
print("+" + "-" * 50 + "+")

iterations = range(1, len(logRegr.precisions) + 1)
plt.plot(iterations, logRegr.precisions, label="Precision")
plt.plot(iterations, logRegr.recalls, label="Recall")
plt.plot(iterations, logRegr.f1_scores, label="F1-Score")
plt.xlabel("Iteration")
plt.ylabel("Metric Value")
plt.title("Training Metrics Over Iterations")
plt.legend()
plt.grid()
plt.show()
