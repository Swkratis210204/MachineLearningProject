# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
from keras.api.datasets import imdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ...existing code...

class MatrixProcessing:
    def __init__(self, top_words=10000, skip_top_words=20, skip_least_frequent=10, start_char=1, oov_char=2, index_from=3):
        self.top_words = top_words
        self.skip_top_words = skip_top_words
        self.skip_least_frequent = skip_least_frequent
        self.index_from = index_from
        self.start_char = start_char
        self.oov_char = oov_char
        self.vocab_to_index = {}
        self.vocab_size = 0

    def load_data(self):
        print("Loading the IMDB dataset...")
        # Load the IMDB dataset
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=self.top_words, skip_top=self.skip_top_words, index_from=self.index_from, 
        )

        return x_train, y_train, x_test, y_test

    def map_reviews_to_binary_vectors(self, reviews):
        binary_vectors = []
        vocab_size = self.top_words  # Adjust vocab size to ignore index_from
    
        for i, review in enumerate(reviews):
            vector = np.zeros(vocab_size, dtype=int)
            for word_index in review:
                adjusted_index = word_index - self.index_from
                if adjusted_index >= 0 and adjusted_index < vocab_size:
                    vector[adjusted_index] = 1
                elif word_index == self.start_char or word_index == self.oov_char:
                    continue  # Ignore start_char and oov_char
            binary_vectors.append(vector)
    
        return binary_vectors

    # def count_word_frequencies_by_vectors(self, binary_vectors):
    #     word_frequencies = {word: 0 for word in self.vocab_to_index.keys()}
    #     index_to_word = {index: word for word, index in self.vocab_to_index.items()}
    #     for vector in binary_vectors:
    #         unique_indices = np.where(vector == 1)[0]
    #         for index in unique_indices:
    #             word_frequencies[index_to_word[index]] += 1
    #     sorted_frequencies = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))
    #     return sorted_frequencies
    
    
    def calculate_information_gain(self, X, y, top_k=1000, n_jobs=-1):
        # Check data format and dimensions
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Calculate mutual information between each feature and labels
        print("Calculating mutual information...")
        mutual_info = mutual_info_classif(X, y, n_jobs=n_jobs)
        print("Mutual information calculated.")
        
        # Get indices of features sorted by information gain (descending order)
        top_feature_indices = np.argsort(mutual_info)[::-1][:top_k]
        print(f"Top feature indices: {top_feature_indices}")
        
        return top_feature_indices, mutual_info[top_feature_indices]