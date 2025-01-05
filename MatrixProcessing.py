# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
from keras.api.datasets import imdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, top_words=10000, skip_top_words=20, skip_least_frequent=10,  start_char=1, oov_char=2, index_from=3):
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
        vocab_size = self.top_words + self.index_from  # Adjust vocab size based on index_from
    
        for i, review in enumerate(reviews):
            vector = np.zeros(vocab_size, dtype=int)
            for word_index in review:
                if word_index >= self.index_from and word_index < vocab_size:
                    vector[word_index] = 1
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