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
from keras.preprocessing.sequence import pad_sequences
from Source import rnnmodel

if __name__ == '__main__':
    n_features = 200 + 1

    model, infenc, infdec = rnnmodel.define_models(n_features, n_features, 400)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    input_data, output_data = rnnmodel.load_data()
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(input_data)

    X1, y = rnnmodel.get_dataset(input_data, output_data, n_features, 10, tokenizer)
    X1_train, X1_test, y_train, y_test = rnnmodel.split_test(X1, y, test_size=0.3)

    X2_train = rnnmodel.decoder_input(y_train)
    X2_test = rnnmodel.decoder_input(y_test)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    history = model.fit([X1_train, X2_train], y_train, epochs=100, batch_size=1, validation_data=([X1_test, X2_test], y_test))

    new_data = []
    with open(os.path.join(os.getcwd(), "Source/predict.txt")) as f:
        new_data.append(f.read())
    tokenizer.fit_on_texts(new_data)
    new_data = tokenizer.texts_to_sequences(new_data)
    X_new = []
    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            X_new.append(to_categorical(new_data[i][j], num_classes=n_features))

    X_new = np.array(X_new)
    X_new = X_new[0:10]
    X_new = X_new.reshape(int(10 / 10), 10, n_features)

    target = rnnmodel.predict_sequence(infenc, infdec, X_new, 10, n_features)

    yhat = [rnnmodel.one_hot_decode(target)]
    text = tokenizer.sequences_to_texts(yhat)
    print(new_data)
    print(text)
 #print(text)