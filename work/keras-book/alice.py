import numpy as np
import codecs

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential

# executed on the parent dir
INPUT_FILE = './data/alice_in_wonderland.txt'

with codecs.open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = [line.strip().lower() for line in f if len(line) != 0]
    text = ' '.join(lines)

chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

SEQLEN = 10
STEP = 1

# for example,
# 'it turned into a pig' ->
# (text        , label)
# ('it turned ', 'i')
# ('t turned i', 'n')
# (' turned in', 't')
# ...
# (' into a pi', 'g')

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

x = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        x[i, j, char2index[ch]] = 1

    y[i, char2index[label_chars[i]]] = 1

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, nb_chars), unroll=True))
model.add(Dense(nb_chars))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# training
for iteration in range(NUM_ITERATIONS):
    print('=' * 50)
    print('Iteration #: {}'.format(iteration))
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)
    test_index = np.random.randint(len(input_chars))
    test_chars = input_chars[test_index]
    print('Generating from seed: {}'.format(test_chars))
    print(test_chars, end='')
    for i in range(NUM_PREDS_PER_EPOCH):
        x_test = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            x_test[0, i, char2index[ch]] = 1
        pred = model.predict(x_test, verbose=0)[0]
        y_pred = index2char[np.argmax(pred)]
        print(y_pred, end='')
        # move forwawrd with test_chars + y_pred
        test_chars = test_chars[1:] + y_pred
    print('')
