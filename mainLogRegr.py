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
filtered_dataset_train, word_to_index = processor.filter_vocabulary(x_train)
filtered_dataset_test, _ = processor.filter_vocabulary(x_test)

# Convert reviews to binary vectors
binary_vectors_train = processor.map_reviews_to_binary_vectors(filtered_dataset_train)
binary_vectors_test = processor.map_reviews_to_binary_vectors(filtered_dataset_test)

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
