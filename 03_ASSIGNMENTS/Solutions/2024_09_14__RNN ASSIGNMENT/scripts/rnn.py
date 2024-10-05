#pip install tensorflow scikit-learn pandas numpy

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class RNNModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
            tf.keras.layers.SimpleRNN(hidden_size, input_shape=(None, input_size)),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def preprocess_data(self, df, target_column):
        
        # Extract relevant columns
        texts = df.iloc[:, 3].values
        labels = df.iloc[:, 2].values

        # Label Encoding the sentiment (Positive, Negative, Neutral)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Tokenizing the text data
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Padding the sequences to ensure equal length
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)
        
        X = padded_sequences
        y = encoded_labels
        # X = df.drop(columns=[target_column]).values
        # y = LabelEncoder().fit_transform(df[target_column].values)
        # Ensure the features are float32 type
        # X = X.astype('float32')
        # X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for RNN input
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, df, target_column, epochs=20, batch_size=32):
        X_train, X_test, y_train, y_test = self.preprocess_data(df, target_column)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    def evaluate(self, df, target_column):
        _, X_test, _, y_test = self.preprocess_data(df, target_column)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# rnn_model = RNNModel(input_size=10, hidden_size=20, output_size=2)
# rnn_model.train(df, target_column='target', epochs=20)
# rnn_model.evaluate(df, target_column='target')

