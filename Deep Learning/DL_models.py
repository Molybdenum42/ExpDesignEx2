import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.layers import Dense, Flatten, Embedding, Attention
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, Layer, GlobalMaxPooling1D
from keras.models import Sequential

from keras import backend as K

def lstm_keras(embed_size, num_classes, embedder):
    model = Sequential()
    model.add(embedder)
    model.add(Dropout(0.25))
    model.add(LSTM(embed_size))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_keras(embed_size, num_classes, embedder):
    model = Sequential()

    model.add(embedder)
    model.add(Dropout(0.25))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.50))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def blstm(embed_size, num_classes, embedder):
    model = Sequential()
    model.add(embedder)
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size)))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def blstm_atten(embed_size, num_classes, embedder):
    model = Sequential()
    model.add(embedder)
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(Attention())
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
