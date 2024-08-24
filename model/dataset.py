import numpy as np
import tensorflow as tf
from .data import greetings


def prepare_dataset():
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(greetings)
    sequences = tokenizer.texts_to_sequences(greetings)
    input_sequences = []

    for seq in sequences:
        for i in range(1, len(seq)):
            input_sequences.append(seq[:i + 1])

    # Pad sequences to make them the same length
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len,
                                                                    padding='pre')

    input_sequences = np.array(input_sequences)
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    return X, y, tokenizer, max_sequence_len
