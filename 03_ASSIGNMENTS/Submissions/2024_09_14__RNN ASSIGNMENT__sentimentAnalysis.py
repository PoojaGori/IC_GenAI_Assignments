import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SentimentAnalysisRNN:
    def __init__(self, input_dim, output_dim, max_len=100):
        """
        Initialize the RNN model.
        :param input_dim: Size of the vocabulary for embedding.
        :param output_dim: Number of output sentiment classes.
        :param max_len: Maximum length of the sequences.
        """
        self.max_len = max_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the vanilla RNN model.
        :return: Compiled RNN model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=64, input_length=self.max_len),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dense(self.output_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_data(self, df, text_column, target_column):
        """
        Preprocess the text data by tokenizing, padding and encoding labels.
        :param df: DataFrame containing the data.
        :param text_column: Name of the column containing the text.
        :param target_column: Name of the column containing the target labels.
        :return: Preprocessed and split data for training and testing.
        """
        # Handle missing values and convert texts to strings
        texts = df[text_column].fillna('').astype(str).values
        labels = df[target_column].values

        # Tokenizing the text data
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.input_dim, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_len, padding='post')

        # Label encoding
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, tokenizer, label_encoder

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the RNN model.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of the training batches.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        :param X_test: Test data.
        :param y_test: Test labels.
        :return: Accuracy, precision, recall, F1-score.
        """
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        return accuracy, precision, recall, f1

    def fine_tune(self, X_train, y_train, X_test, y_test, learning_rate=0.001):
        """
        Fine-tune the model by adjusting hyperparameters like learning rate.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :param y_test: Test labels.
        :param learning_rate: Learning rate for optimization.
        """
        # Recompile the model with the new learning rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

        # Retrain the model
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


def load_data_rnn( file_name):
        try:
            df = pd.read_csv(f'../data/{file_name}', names=['col0', 'col1', 'sentiment', 'text'])
            print("Data loaded successfully.")
            return df
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
# Example usage:
# Load your dataset here
# df = pd.read_csv('your_dataset.csv')
file_name = 'twitter_training.csv'
df = load_data_rnn(file_name)
df = df.dropna()

# Initialize and preprocess the data
rnn = SentimentAnalysisRNN(input_dim=5000, output_dim=4)  # Adjust 'output_dim' based on the number of sentiment classes
X_train, X_test, y_train, y_test, tokenizer, label_encoder = rnn.preprocess_data(df, text_column='text', target_column='sentiment')

# Train the model
rnn.train(X_train, y_train, epochs=10)

# Evaluate the model
rnn.evaluate(X_test, y_test)

# Fine-tune the model (optional)
rnn.fine_tune(X_train, y_train, X_test, y_test, learning_rate=0.0005)

# Evaluate the model
rnn.evaluate(X_test, y_test)
