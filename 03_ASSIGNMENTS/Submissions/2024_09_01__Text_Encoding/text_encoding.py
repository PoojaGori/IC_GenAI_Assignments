# text_encoding.py

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors
import re

class TextEncoding:
    def __init__(self, data):
        self.data = data
        self.processed_text = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = nltk.word_tokenize(text)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    def prepare_data(self):
        self.data['processed_text'] = self.data['text_column'].apply(self.preprocess_text)

    def bag_of_words(self):
        vectorizer_bow = CountVectorizer()
        bow_matrix = vectorizer_bow.fit_transform(self.data['processed_text'])
        self.bow_vectorizer = vectorizer_bow
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer_bow.get_feature_names_out())
        return bow_df

    def tf_idf(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['processed_text'])
        self.tfidf_vectorizer = tfidf_vectorizer
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        return tfidf_df

    def display_vocabulary(self, method='bow'):
        if method == 'bow' and hasattr(self, 'bow_vectorizer'):
            vocab = self.bow_vectorizer.vocabulary_
            print("\nVocabulary (Bag of Words):")
            for word, idx in sorted(vocab.items(), key=lambda item: item[1]):
                print(f"{word}: {idx}")
        elif method == 'tfidf' and hasattr(self, 'tfidf_vectorizer'):
            vocab = self.tfidf_vectorizer.vocabulary_
            print("\nVocabulary (TF-IDF):")
            for word, idx in sorted(vocab.items(), key=lambda item: item[1]):
                print(f"{word}: {idx}")
        else:
            print(f"Vocabulary for method '{method}' is not available or has not been generated yet.")

    def train_word2vec(self, vector_size=100, window=5, min_count=2):
        tokenized_text = self.data['processed_text'].apply(nltk.word_tokenize)
        w2v_model = Word2Vec(sentences=tokenized_text, vector_size=vector_size, window=window, min_count=min_count)
        return w2v_model

    def load_glove_embeddings(self, glove_path):
        glove_embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = vector
        return glove_embeddings

    def train_fasttext(self, vector_size=100, window=5, min_count=2):
        tokenized_text = self.data['processed_text'].apply(nltk.word_tokenize)
        fasttext_model = FastText(sentences=tokenized_text, vector_size=vector_size, window=window, min_count=min_count)
        return fasttext_model
