import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer


class SentimentAnalysis:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(f'../data/{self.file_name}')
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"File {self.file_name} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def eda(self):
        if self.data is not None:
            print("Exploratory Data Analysis:")
            print(self.data.head())
            print(self.data.info())
            print(self.data.describe())
        else:
            print("No data to perform EDA on.")   
            return 
           
        training_df = self.data    
        # Extract relevant columns
        texts = training_df.iloc[:, 3].values
        labels = training_df.iloc[:, 2].values

        # Label Encoding the sentiment (Positive, Negative, Neutral)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Tokenizing the text data
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Padding the sequences to ensure equal length
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)
        
        return padded_sequences, encoded_labels

        # Now padded_sequences and encoded_labels can be used for model training        

    def apply_rnn(self):
        if self.data is None:
            print("No data to apply RNN on.")
            return
        
        print("Applying RNN...")
        padded_sequences, encoded_labels = self.eda()
        
        # Implement RNN model training and evaluation here
        # Define the RNN Model architecture
        model = Sequential()

        # Embedding layer: This layer is responsible for converting words into vectors of fixed size
        model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))

        # SimpleRNN layer: This is the core of the vanilla RNN
        model.add(SimpleRNN(64, return_sequences=False))

        # Output layer: Fully connected layer with one neuron for each sentiment class
        model.add(Dense(3, activation='softmax'))  # Assuming 3 sentiment classes: Positive, Negative, Neutral

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Summary of the model
        model.summary()

        # Train the model
        history = model.fit(padded_sequences, encoded_labels, epochs=10, batch_size=32, validation_split=0.2)

        # After training, evaluate the model
        loss, accuracy = model.evaluate(padded_sequences, encoded_labels)
        print(f"Training Accuracy: {accuracy * 100:.2f}%")
        
        

# # Example usage
# if __name__ == "__main__":
#     file_name = 'twitter_training.csv'
#     sentiment_analysis = SentimentAnalysis(file_name)
#     sentiment_analysis.load_data()
    