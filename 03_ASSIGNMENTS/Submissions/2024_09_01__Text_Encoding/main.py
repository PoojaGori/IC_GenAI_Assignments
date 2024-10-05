# main.py

import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from text_encoding import TextEncoding

# Download NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

def load_movie_reviews():
    """Load the IMDb movie reviews dataset and return a DataFrame."""
    reviews = []
    for fileid in movie_reviews.fileids():
        category = movie_reviews.categories(fileid)[0]
        review_text = movie_reviews.raw(fileid)
        reviews.append({'text_column': review_text, 'category': category})
    return pd.DataFrame(reviews)

def main():
    # Load the IMDb movie reviews dataset
    df = load_movie_reviews()
    
    df = df.sample(n=10, random_state=42).reset_index(drop=True)  # Shuffle the dataset

    # Initialize the TextEncoding class with the dataset
    encoder = TextEncoding(df)

    # Preprocess the text data
    encoder.prepare_data()

    # Bag of Words
    bow_df = encoder.bag_of_words()
    print("Bag of Words (BoW):")
    print(bow_df.head())

    # Display Bag of Words vocabulary
    # encoder.display_vocabulary(method='bow')

    # TF-IDF
    tfidf_df = encoder.tf_idf()
    print("\nTF-IDF:")
    print(tfidf_df.head())

    # Display TF-IDF vocabulary
    encoder.display_vocabulary(method='tfidf')

    # Word2Vec
    w2v_model = encoder.train_word2vec()
    print("\nWord2Vec model trained. Vector for 'movie':")
    print(w2v_model.wv['movie'])  # Example word from the dataset

    # # GloVe (update with your GloVe path)
    # glove_path = 'glove.6B.100d.txt'
    # glove_embeddings = encoder.load_glove_embeddings(glove_path)
    # print("\nGloVe embedding for 'movie':")
    # print(glove_embeddings.get('movie'))  # Example word from the dataset

    # FastText
    fasttext_model = encoder.train_fasttext()
    print("\nFastText model trained. Vector for 'movie':")
    print(fasttext_model.wv['movie'])  # Example word from the dataset

if __name__ == '__main__':
    main()
