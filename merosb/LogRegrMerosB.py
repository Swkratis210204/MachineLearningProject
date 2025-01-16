import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MatrixProcessing import MatrixProcessing

print("Starting the model now.")
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

# Shuffle training data for learning curves
binary_vectors_train, y_train = shuffle(binary_vectors_train, y_train, random_state=42)

# Match the number of iterations
custom_iterations = 1000  # Replace with your implementation's iteration count

# Initialize Logistic Regression
LogRegr = LogisticRegression(penalty='l1', solver='saga', max_iter=custom_iterations, C=1.0, verbose=0)

# Add Learning Curves
training_sizes = np.linspace(0.1, 1.0, 10)  # 10 training size steps
precision_train, recall_train, f1_train = [], [], []
precision_dev, recall_dev, f1_dev = [], [], []

# Incremental training
for frac in training_sizes:
    n_samples = int(frac * binary_vectors_train.shape[0])
    x_train_subset = binary_vectors_train[:n_samples]
    y_train_subset = y_train[:n_samples]
    
    # Train model
    LogRegr.fit(x_train_subset, y_train_subset)
    
    # Predict on training subset
    y_pred_train = LogRegr.predict(x_train_subset)
    train_precision = precision_score(y_train_subset, y_pred_train, pos_label=1)
    train_recall = recall_score(y_train_subset, y_pred_train, pos_label=1)
    train_f1 = f1_score(y_train_subset, y_pred_train, pos_label=1)
    precision_train.append(train_precision)
    recall_train.append(train_recall)
    f1_train.append(train_f1)
    
    # Predict on development data
    y_pred_dev = LogRegr.predict(binary_vectors_test)
    dev_precision = precision_score(y_test, y_pred_dev, pos_label=1)
    dev_recall = recall_score(y_test, y_pred_dev, pos_label=1)
    dev_f1 = f1_score(y_test, y_pred_dev, pos_label=1)
    precision_dev.append(dev_precision)
    recall_dev.append(dev_recall)
    f1_dev.append(dev_f1)

    # Print results for each subset
    print(f"Training with {n_samples} examples:")
    print(f"Train Precision: {train_precision:.5f}, Train Recall: {train_recall:.5f}, Train F1-Score: {train_f1:.5f}")
    print(f"Dev Precision: {dev_precision:.5f}, Dev Recall: {dev_recall:.5f}, Dev F1-Score: {dev_f1:.5f}")
    print("")

# Final evaluation on development set
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
print(f"Final Accuracy: {accuracy:.5f}")
print("\nClass-Specific Metrics:")
for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
    print(f"Class {i}: Precision: {p:.5f}, Recall: {r:.5f}, F1-Score: {f:.5f}")

print("\nAverages:")
print(f"Precision (Macro): {precision_macro:.5f}, Precision (Micro): {precision_micro:.5f}")
print(f"Recall (Macro): {recall_macro:.5f}, Recall (Micro): {recall_micro:.5f}")
print(f"F1-Score (Macro): {f1_macro:.5f}, F1-Score (Micro): {f1_micro:.5f}")

# Plot Learning Curves
plt.figure(figsize=(10, 6))
plt.plot(training_sizes * len(binary_vectors_train), precision_train, label='Precision (Train)', color='blue')
plt.plot(training_sizes * len(binary_vectors_train), recall_train, label='Recall (Train)', color='orange')
plt.plot(training_sizes * len(binary_vectors_train), f1_train, label='F1-Score (Train)', color='green')

plt.plot(training_sizes * len(binary_vectors_train), precision_dev, '--', label='Precision (Dev)', color='blue')
plt.plot(training_sizes * len(binary_vectors_train), recall_dev, '--', label='Recall (Dev)', color='orange')
plt.plot(training_sizes * len(binary_vectors_train), f1_dev, '--', label='F1-Score (Dev)', color='green')

plt.title("Training Metrics Over Increasing Training Set Size")
plt.xlabel("Number of Training Examples")
plt.ylabel("Metric Value")
plt.legend()
plt.grid()
plt.show()
