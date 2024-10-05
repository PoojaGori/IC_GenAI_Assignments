import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class LSTMNextWordPredictor:
    def __init__(self, max_vocab_size=5000, sequence_length=50):
        self.max_vocab_size = max_vocab_size
        self.sequence_length = sequence_length
        self.tokenizer = None
        self.model = None

    def preprocess_text(self, text):
        """
        Preprocess the raw text data: tokenize and prepare sequences for LSTM.
        :param text: Raw text input.
        :return: Tokenized and padded sequences, word index.
        """
        # Initialize the tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([text])

        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences([text])[0]

        # Create input sequences and labels
        input_sequences = []
        labels = []
        for i in range(1, len(sequences)):
            n_gram_sequence = sequences[:i+1]
            input_sequences.append(n_gram_sequence[:-1])
            labels.append(n_gram_sequence[-1])

        # Pad sequences to ensure uniform length
        input_sequences = pad_sequences(input_sequences, maxlen=self.sequence_length, padding='pre')

        return np.array(input_sequences), np.array(labels)
    
    def preprocess_text_limited(self, text):
        """
        Preprocess the raw text data: tokenize and prepare sequences for LSTM.
        Limit the number of sequences to avoid memory issues.
        :param text: Raw text input.
        :return: Tokenized and padded sequences, word index.
        """
        # Tokenize the text
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([text])
        sequences = self.tokenizer.texts_to_sequences([text])[0]

        input_sequences = []
        labels = []
        
        # Limit the number of sequences to avoid memory error
        for i in range(1, min(len(sequences), self.max_sequences)):
            n_gram_sequence = sequences[:i+1]
            input_sequences.append(n_gram_sequence[:-1])
            labels.append(n_gram_sequence[-1])
            
            if len(input_sequences) >= self.max_sequences:  # Stop if max sequences are reached
                break

        # Pad sequences
        input_sequences = pad_sequences(input_sequences, maxlen=self.sequence_length, padding='pre')
        
        return np.array(input_sequences), np.array(labels)

    def build_model(self, input_dim, output_dim, embedding_dim=64, lstm_units=64):
        """
        Build and compile the LSTM model.
        :param input_dim: Vocabulary size.
        :param output_dim: Number of classes (vocabulary size).
        :param embedding_dim: Dimension of the embedding layer.
        :param lstm_units: Number of units in the LSTM layer.
        :return: Compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=self.sequence_length),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the LSTM model.
        :param X_train: Training input sequences.
        :param y_train: Training labels.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict_next_word(self, input_text):
        """
        Predict the next word for a given input text.
        :param input_text: The input text sequence for prediction.
        :return: Predicted next word.
        """
        tokenized_input = self.tokenizer.texts_to_sequences([input_text])[0]
        tokenized_input = pad_sequences([tokenized_input], maxlen=self.sequence_length, padding='pre')
        predicted_id = np.argmax(self.model.predict(tokenized_input), axis=-1)
        predicted_word = self.tokenizer.index_word.get(predicted_id[0], "<OOV>")
        return predicted_word

# Load the text file
with open(r'D:\POOJA\repos\Courses\GenAI\BE_InnerCircle\03_ASSIGNMENTS\Solutions\LSTM\data\LSTM_DATA.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Initialize the model
lstm_predictor = LSTMNextWordPredictor(max_vocab_size=5000, sequence_length=50, max_sequences=10000)

# Preprocess the text data
X, y = lstm_predictor.preprocess_text_limited(text_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
vocab_size = lstm_predictor.tokenizer.num_words
lstm_predictor.build_model(input_dim=vocab_size, output_dim=vocab_size)

# Train the model
lstm_predictor.train(X_train, y_train, epochs=10, batch_size=64)

# Example of next-word prediction
input_text = "Once upon a time there was a"
predicted_word = lstm_predictor.predict_next_word(input_text)
print(f"Predicted next word: {predicted_word}")
