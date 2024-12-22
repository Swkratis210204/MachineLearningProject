from MatrixProcessing import MatrixProcessing

# Initialize processor
processor = MatrixProcessing(top_words=10000, skip_top_words=20, skip_least_frequent=10)

# Fortose kai filtrare dedomena
x_train, y_train, x_test, y_test = processor.load_data()
filtered_dataset, LekseisKaiIndex = processor.filter_vocabulary(x_train)

#Binary vector gia kathe review
binary_vectors=processor.map_reviews_to_binary_vectors(filtered_dataset)

#Sixnotita leksewn gia kathe review, an mia leksi emfanizetai >1 fores se ena review, krataw 1
word_frequencies = processor.count_word_frequencies_by_vectors(binary_vectors)















