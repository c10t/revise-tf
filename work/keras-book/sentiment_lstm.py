import codecs
import os

import nltk
import numpy as np


from collections import Counter

from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

DATA_DIR = './data'
LOG_DIR = './logs'
TRAIN = 'umich-sentiment-train.txt'

maxlen = 0
word_freqs = Counter()
num_recs = 0

with codecs.open(os.path.join(DATA_DIR, TRAIN), 'r', 'utf-8') as ftrain:
    for line in ftrain:
        label, sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        maxlen = max(maxlen, len(words))
        for word in words:
            word_freqs[word] += 1

        num_recs += 1

print(maxlen)
print(len(word_freqs))
