from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import text_to_word_sequence
import os
import json

class Model:    
    def __init__(self):
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
        self.model_weights_path = os.path.join(MODELS_DIR, 'sentiment_weights.h5')
        self.words_index_path = os.path.join(MODELS_DIR, 'word_index.json')
        self.word_index = self.load_dictionary()
        self.model = self.load()

    def load_data(self):
        imdb = keras.datasets.imdb
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        return (train_data, train_labels), (test_data, test_labels)

    def load_dictionary(self):
        word_index = None
        try:
            with open(self.words_index_path, 'r') as json_file:
                word_index = json.load(json_file)
        except:
            imdb = keras.datasets.imdb
            word_index = imdb.get_word_index()
            with open(self.words_index_path, 'w') as json_file:
                json.dump(word_index, json_file)

        # The first indices are reserved
        word_index = {k:(v+3) for k,v in word_index.items()} 
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        return word_index
    
    def preprocess_data(self, train_data, test_data):
        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=self.word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,   
                                                       value=self.word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
        
        return train_data, test_data
    
    def make_compiled_model(self):
        vocab_size = 10000
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return model

    def train_model(self, train_data, train_labels, model):
        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]

        model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
    
    def save_model(self, model):
        model.save(self.model_weights_path)

    def load(self):
        try:
            model = self.make_compiled_model()
            model.load_weights(self.model_weights_path)
            return model
        except:
            (train_data, train_labels), (test_data, _) = self.load_data()
            train_data, test_data = self.preprocess_data(train_data, test_data)
            model = self.make_compiled_model()
            self.train_model(train_data, train_labels, model)
            self.save_model(model)
            return model

    def predict(self, input_sentence):
        tokens = text_to_word_sequence(input_sentence)
        input_data = []
        for word in tokens:
            for (key, value) in self.word_index.items():
                if key == word: input_data.append(value)

        input_data = keras.preprocessing.sequence.pad_sequences([input_data],
                                                       value=self.word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
        try:
            predictions = self.model.predict([input_data])
            return predictions
        except:
            return None
