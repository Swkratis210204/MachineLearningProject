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
    
    # def count_word_frequencies_by_vectors(self, binary_vectors):
    #     word_frequencies = {word: 0 for word in self.vocab_to_index.keys()}
    #     index_to_word = {index: word for word, index in self.vocab_to_index.items()}
    #     for vector in binary_vectors:
    #         unique_indices = np.where(vector == 1)[0]
    #         for index in unique_indices:
    #             word_frequencies[index_to_word[index]] += 1
    #     sorted_frequencies = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))
    #     return sorted_frequencies
    

    def calculate_information_gain(self, binary_vectors, labels, top_k=1000, batch_size=1000):
        binary_vectors = np.array(binary_vectors)
        vocab_size = binary_vectors.shape[1]
        info_gain = np.zeros(vocab_size)
        
        for start in range(0, vocab_size, batch_size):
            end = min(start + batch_size, vocab_size)
            info_gain[start:end] = mutual_info_classif(
                binary_vectors[:, start:end], labels, discrete_features=True
            )
        
        index_to_word = {index: word for word, index in self.vocab_to_index.items()}
        word_info_gain = {index_to_word[idx]: gain for idx, gain in enumerate(info_gain) if idx in index_to_word}
        sorted_info_gain = dict(sorted(word_info_gain.items(), key=lambda item: item[1], reverse=True))
        top_words = list(sorted_info_gain.items())[:top_k]
        
        return top_words

