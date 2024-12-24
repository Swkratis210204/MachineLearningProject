import numpy as np
from MatrixProcessing import MatrixProcessing
from LogisticRegression import LogisticRegression

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
logRegr = LogisticRegression(lr=0.01, n_iter=1000, tolerance=1e-6)
logRegr.fit(binary_vectors_train, y_train, batch_size=256)
y_pred = logRegr.predict(binary_vectors_test)

# Calculate accuracy
accuracy = logRegr.accuracy(y_pred, y_test)
print(f"Logistic Regression Accuracy: {accuracy}")
