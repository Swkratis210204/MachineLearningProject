from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from MatrixProcessing import MatrixProcessing

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

# Match the number of iterations
custom_iterations = 1000  # Replace with your implementation's iteration count

# Train Logistic Regression
LogRegr = LogisticRegression(penalty='l1', solver='saga', max_iter=custom_iterations, C=1.0, verbose=0)
LogRegr.fit(binary_vectors_train, y_train)

# Evaluate model
predictions = LogRegr.predict(binary_vectors_test)
accuracy = accuracy_score(y_test, predictions)

# Class-specific metrics
precision_per_class = precision_score(y_test, predictions, average=None)
recall_per_class = recall_score(y_test, predictions, average=None)
f1_per_class = f1_score(y_test, predictions, average=None)

# Macro and Micro averages
precision_macro = precision_score(y_test, predictions, average='macro')
precision_micro = precision_score(y_test, predictions, average='micro')

recall_macro = recall_score(y_test, predictions, average='macro')
recall_micro = recall_score(y_test, predictions, average='micro')

f1_macro = f1_score(y_test, predictions, average='macro')
f1_micro = f1_score(y_test, predictions, average='micro')

# Print results
print(f"Accuracy: {accuracy:.5f}")
print("\nClass-Specific Metrics:")
for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
    print(f"Class {i}: Precision: {p:.5f}, Recall: {r:.5f}, F1-Score: {f:.5f}")

print("\nAverages:")
print(f"Precision (Macro): {precision_macro:.5f}, Precision (Micro): {precision_micro:.5f}")
print(f"Recall (Macro): {recall_macro:.5f}, Recall (Micro): {recall_micro:.5f}")
print(f"F1-Score (Macro): {f1_macro:.5f}, F1-Score (Micro): {f1_micro:.5f}")