from MatrixProcessing import MatrixProcessing

# Initialize processor
processor = MatrixProcessing(top_words=10000, skip_top_words=20, skip_least_frequent=10)

# Fortose kai filtrare dedomena
x_train, y_train, x_test, y_test = processor.load_data()
filtered_dataset_train, word_to_index = processor.filter_vocabulary(x_train)
filtered_dataset_test, _ = processor.filter_vocabulary(x_test)

#Binary vector gia kathe review
binary_vectors_train = processor.map_reviews_to_binary_vectors(filtered_dataset_train)
binary_vectors_test = processor.map_reviews_to_binary_vectors(filtered_dataset_test)


top_words = processor.calculate_information_gain(binary_vectors_train, y_train, top_k=1000)













