from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf
from .dataset import prepare_dataset


class Model:
    _tokenizer = None
    _model = None
    _max_sequence_length = None

    def __init__(self):
        dataset_attr = prepare_dataset()
        # X, y, tokenizer, max_sequence_len
        self._tokenizer = dataset_attr[2]

        self._max_sequence_length = dataset_attr[3]
        self._X = dataset_attr[0]
        self._y = dataset_attr[1]

    def create_model(self):
        model = Sequential()
        model.add(Embedding(len(self._tokenizer.word_index) + 1, 10, input_length=self._max_sequence_length - 1))
        model.add(LSTM(50))
        model.add(Dense(len(self._tokenizer.word_index) + 1, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        self._model = model

        return self

    def train_model(self):
        self._model.fit(self._X, self._y, epochs=500, verbose=1)

        self.save_model('./model.h5')

        return self


    def save_model(self, filepath):
        if self._model:
            self._model.save(filepath)
        else:
            raise ValueError("Model has not been created or trained.")



class TrainedModel:
    _model: None

    def __init__(self):
        pass

    def load_model(self):
        self._model = load_model('./model.h5')
        dataset_attr = prepare_dataset()
        # X, y, tokenizer, max_sequence_len
        self._tokenizer = dataset_attr[2]
        self._max_sequence_length = dataset_attr[3]

        return self

    def generate_greeting(self, seed_text, next_words=20):
        for _ in range(next_words):
            token_list = self._tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=self._max_sequence_length - 1,
                                                                       padding='pre')
            predicted = self._model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=-1)[0]
            predicted_word = self._tokenizer.index_word[predicted_word_index]
            seed_text += predicted_word
        return seed_text

