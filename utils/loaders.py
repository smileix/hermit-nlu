# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_word_embeddings(path):
    print('Loading word embeddings...')
    loaded = np.load(path)
    embedding_matrix = loaded['embedding']
    word2idx = dict()
    for i, word in enumerate(loaded['word2idx']):
        word2idx[word] = i
    del loaded
    return embedding_matrix, word2idx


def load_label_encoder(path):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(path)
    return encoder
