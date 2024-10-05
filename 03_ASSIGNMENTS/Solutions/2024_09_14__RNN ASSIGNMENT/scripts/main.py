
import pandas as pd
from rnn import RNNModel

# from sentiment_analysis import SentimentAnalysis


def load_data_rnn( file_name):
        try:
            df = pd.read_csv(f'../data/{file_name}', names=['col0', 'col1', 'target', 'text'])
            print("Data loaded successfully.")
            return df
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
            

def main():
    
    file_name = 'twitter_training.csv'
    # sentiment_analysis = SentimentAnalysis(file_name)
    # sentiment_analysis.load_data()
    df = load_data_rnn(file_name)
        
    rnn_model = RNNModel(input_size=10, hidden_size=20, output_size=2)
    
    X_train, X_test, y_train, y_test =  rnn_model.preprocess_data(df, target_column='target')
    
    rnn_model.train(df, target_column='target', epochs=20)
    rnn_model.evaluate(df, target_column='target')
    
    
    
    


if __name__ == '__main__':
    main()