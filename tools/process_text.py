import nltk
import string
import numpy as np
import streamlit as st
from collections import Counter
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')


@st.cache_resource
def load_word2vec_models():
    try:
        cbow_model = KeyedVectors.load_word2vec_format('data/model/cbow_ptbr_100d/cbow_s100.txt')
        skipgram_model = KeyedVectors.load_word2vec_format('data/model/skipgram_ptbr_100d/skip_s100.txt')
        return cbow_model, skipgram_model
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None


class ProcessText:
    def __init__(self):
        download_nltk_resources()
        self.stop_words = set(nltk.corpus.stopwords.words('portuguese'))
        self.cbow_model, self.skipgram_model = load_word2vec_models()


    # Função para pré-processar e tokenizar texto
    def preprocess_text(self, text):
        # Tokenização e remoção de pontuação
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        # Remoção de stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    # Função para vetorizar uma frase usando o modelo Word2Vec
    def sentence_to_vector(self, sentence, model):
        tokens = self.preprocess_text(sentence)
        # Filtra tokens presentes no vocabulário do modelo
        valid_tokens = [token for token in tokens if token in model.key_to_index]
        
        if len(valid_tokens) > 1:
            # Contador de frequência simples para dar mais peso às palavras menos comuns na frase
            token_counts = Counter(valid_tokens)
            total = sum(token_counts.values())
            weights = [1.0 / (token_counts[token] / total) for token in valid_tokens]
            vectors = [model[token] * weight for token, weight in zip(valid_tokens, weights)]
            return np.sum(vectors, axis=0) / sum(weights)
        
    # Função para extrair palavras-chave usando TF-IDF
    def extract_keywords(self, sentences, n_keywords=5):
        if not sentences:
            return []
        
        vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=self.preprocess_text)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Soma dos TF-IDF por palavra
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        
        # Obtém as palavras com maior soma de TF-IDF
        top_indices = tfidf_sums.argsort()[-n_keywords:][::-1]
        return [feature_names[i] for i in top_indices]

    # Função para extrair palavras-chave baseada em frequência
    def extract_keywords_by_frequency(self, sentences, n_keywords=5):
        if not sentences:
            return []
        
        # Tokeniza e pré-processa todas as frases
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(self.preprocess_text(sentence))
        
        # Conta frequência de cada palavra
        word_freq = Counter(all_tokens)
        
        # Retorna as n palavras mais frequentes
        return [word for word, _ in word_freq.most_common(n_keywords)]