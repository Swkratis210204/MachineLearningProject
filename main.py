from MatrixProcessing import MatrixProcessing

# Initialize processor
processor = MatrixProcessing(top_words=10000, skip_top_words=20, skip_least_frequent=10)

# Load and filter data
x_train, y_train, x_test, y_test = processor.load_data()
filtered_dataset, LekseisKaiIndex = processor.filter_vocabulary(x_train)

binary_vectors=processor.map_reviews_to_binary_vectors(filtered_dataset)















