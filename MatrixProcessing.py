import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
from keras.api.datasets import imdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, top_words=10000, skip_top_words=20, skip_least_frequent=10, index_from=3):
        self.skip_top_words = skip_top_words
        self.index_from = index_from
        self.vocab_size = top_words-skip_least_frequent
        print("Vocab size: ", self.vocab_size)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=self.vocab_size,
            skip_top=self.skip_top_words,
            index_from=self.index_from
        )
        return x_train, y_train, x_test, y_test

    def map_reviews_to_binary_vectors(self, dataset):
        binary_vectors = []
        for review in dataset:
            vector = np.zeros(self.vocab_size, dtype=int)
            for word_index in review:
                if word_index >= self.index_from:
                    vector[word_index - self.index_from] = 1
            binary_vectors.append(vector)
        return binary_vectors


    def calculate_information_gain(self, binary_vectors, labels, top_k=1000):
        # Calculate mutual information for each feature
        mutual_info = mutual_info_classif(binary_vectors, labels, discrete_features=True)
        
        # Get the indices of the top_k features with the highest mutual information
        top_indices = np.argsort(mutual_info)[-top_k:]
        
        # Sort the indices in descending order of mutual information
        top_indices = top_indices[::-1]
        
        return top_indices