# -*- coding: UTF-8 -*-
import tarfile
import numpy as np
import os
from progress.bar import Bar

np.random.seed(42)


def glove_embeddings(input_file, output_file, include_empty_char=True, padding=True):
    embedding_matrix = []
    word2idx = []
    embedding_dim = 300
    print('Unpacking {}...'.format(input_file))
    tar = tarfile.open(input_file, "r:gz")
    print('done!')
    for member in tar.getmembers():
        f = tar.extractfile(member)
        first_line = f.readline()
        embedding_dim = len([x.strip() for x in first_line.split()]) - 1
        if padding:
            word2idx.append('AAA_PADDING')
            embedding_matrix.append(np.random.rand(embedding_dim))
        data = [x.strip().lower() for x in first_line.split()]
        word = data[0]
        word2idx.append(word)
        embedding_matrix.append(np.asarray(data[1:embedding_dim + 1], dtype='float32'))
        bar = Bar('Loading Word Embeddings: ')
        for idx, line in enumerate(f):
            bar.next()
            try:
                data = [x.strip().lower() for x in line.split()]
                word = data[0]
                word2idx.append(word)
                embedding_matrix.append(np.asarray(data[1:embedding_dim + 1], dtype='float32'))
            except Exception as e:
                print('Exception occurred in `load_glove_embeddings`:', e)
                continue
    if include_empty_char:
        word2idx.append('')
        embedding_matrix.append(np.random.rand(embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix)
    word2idx = np.asarray(word2idx)
    print('Saving word embeddings...'),
    np.savez_compressed(os.path.join(os.path.dirname(input_file), output_file), embedding=embedding_matrix,
                        word2idx=word2idx)
    print('done!')
