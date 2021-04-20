import os

import numpy as np
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def load_data():
    X, Y = [], []
    input_path = os.path.join(os.getcwd(), "Input")
    output_path = os.path.join(os.getcwd(), "Output")

    for file in os.listdir(input_path):
        with open(os.path.join(input_path, file)) as f:
            X.append(f.read())
        with open(os.path.join(output_path, file)) as f:
            Y.append(f.read())
    return X, Y


def get_dataset(X, Y, n_features, lenght):
    X = tokenizer.texts_to_sequences(X)
    Y = tokenizer.texts_to_sequences(Y)

    y = []
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            y.append(to_categorical(Y[i][j], num_classes=n_features))
    y = np.array(y)
    y = y[0:lenght]

    X1 = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            X1.append(to_categorical(X[i][j], num_classes=n_features))
    X1 = np.array(X1)
    X1 = X1[0:lenght]

    X1 = X1.reshape(int(lenght/10), 10, n_features)
    y = y.reshape(int(lenght/10), 10, n_features)
    return X1, y

def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

def decoder_input(input):
    X2 = []
    for i in range(len(input)):
        array = np.zeros((10, 201))
        array[1:10] = input[i][:-1]
        X2.append(array)
    X2 = np.array(X2)

    return X2

def split_test(X1, y, test_size):

    split = int(len(X1)*test_size)
    split2 = int(len(X1))

    X1_train = X1[0:split2]
    X1_test = X1[split:len(X1)]

    y_train = y[0:split2]
    y_test = y[split:len(y)]

    return X1_train, X1_test, y_train, y_test

if __name__ == '__main__':
    n_features = 200 + 1

    train, infenc, infdec = define_models(n_features, n_features, 300)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    input_data, output_data = load_data()
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(input_data)

    X1, y = get_dataset(input_data, output_data, n_features, 10)
    X1_train, X1_test, y_train, y_test = split_test(X1, y, test_size=0.2)

    X2_train = decoder_input(y_train)
    X2_test = decoder_input(y_test)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    history = train.fit([X1_train, X2_train], y_train, epochs=200, batch_size=1, validation_data=([X1_test, X2_test], y_test))

    prediction = train.predict()