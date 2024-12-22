# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
from tensorflow.keras.datasets import imdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, top_words=10000, skip_top_words=20, skip_least_frequent=10, index_from=3):
        self.top_words = top_words
        self.skip_top_words = skip_top_words
        self.skip_least_frequent = skip_least_frequent
        self.index_from = index_from
        self.vocab_to_index = {}
        self.vocab_size = 0

    def load_data(self):
        """
        Loads the IMDB dataset with the specified parameters.
        """
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=self.top_words,
            skip_top=self.skip_top_words,
            index_from=self.index_from
        )
        return x_train, y_train, x_test, y_test

    def filter_vocabulary(self, dataset):
        word_index = imdb.get_word_index()
        word_index = {word: idx for word, idx in word_index.items() if word.isascii()}

        # Filter vocabulary based on index constraints
        filtered_vocab = {
            word: idx for word, idx in word_index.items()
            if self.skip_top_words <= idx < self.top_words + self.index_from
        }

        # Create vocab_to_index as word: index mapping
        self.vocab_to_index = {word: idx for word, idx in filtered_vocab.items()}
        self.vocab_size = len(self.vocab_to_index)

        # Convert valid indices to a set for faster lookups
        valid_word_indices = set(self.vocab_to_index.values())

        # Filter the dataset to include only words in the valid vocabulary
        filtered_dataset = [
            [word for word in review if word in valid_word_indices]
            for review in dataset
        ]

        return filtered_dataset, self.vocab_to_index

    def map_reviews_to_binary_vectors(self, filtered_dataset):
        binary_vectors = []
        for review in filtered_dataset:
            vector = np.zeros(self.vocab_size, dtype=int)
            for word_index in review:
                if 0 <= word_index < self.vocab_size:
                    vector[word_index] = 1
            binary_vectors.append(vector)
        return binary_vectors


