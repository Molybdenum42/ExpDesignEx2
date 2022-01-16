import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, Bidirectional
from keras.models import Sequential

from keras import backend as K
from keras.engine.topology import Layer, InputSpec

# add this bit to each dataset's workbook
random_embedder = Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True)
glove_embedding = Embedding(vocab_size, embed_size,
                            embeddings_initializer=keras.initializers.Constant(glove_embedding_matrix), trainable=False)
word2vec_embedding = Embedding(vocab_size, embed_size,
                               embeddings_initializer=keras.initializers.Constant(word2vec_embedding_matrix),
                               trainable=False)


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

    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(35))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
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


class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def blstm_atten(embed_size, num_classes, embedder):
    model = Sequential()
    model.add(embedder)
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(AttLayer())
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
