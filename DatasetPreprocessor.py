import csv
import math
import re
import string
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import decomposition, svm
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wordcloud import STOPWORDS, WordCloud
from utils import slice_data_frame

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def strip_strange_symbols(text):
    return re.sub(r'[\W_]+', ' ', text)

# Removing URLs
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

# Doing stemming
def lemmatize(text):
    word_filtered = []
    for i in text.split():
        word_tokens = word_tokenize(i)
        for w in word_tokens:
            w = lemmatizer.lemmatize(w, pos='v')
            word_filtered.append(w)
    return " ".join(w for w in word_filtered)

# Removing noisy text
def denoise_text(text):
    text = strip_html(text)
    text = strip_strange_symbols(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

# perform principal component analysis
def principal_component_analysis(X):
    print(f'Before:\tn_samples = {len(X)}, n_features = {len(X[0])}')
    if (len(X) < len(X[0])):
        pca = decomposition.PCA(n_components=len(X),svd_solver='full')
    else:
        pca = decomposition.PCA(n_components='mle',svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    print(f'After:\tn_samples = {len(X)}, n_features = {len(X[0])}')
    print(f'Score (cumulative variance) = {np.cumsum(pca.explained_variance_ratio_)[-1]}')
    return X

def load_and_preprocess(min_df,n_news,shuffle):
    nltk.download('stopwords',quiet=True)
    nltk.download('punkt',quiet=True)
    nltk.download('wordnet',quiet=True)

    try:
        # Tenta carregar os data sets ja processado do disco
        print('Attempting to load data from cached data set...')
        X = np.load(f'./assets/cache_npz/X_{n_news}.npz')
        X = X.f.arr_0
        Y = np.load(f'./assets/cache_npz/Y_{n_news}.npz')
        Y = Y.f.arr_0
        print('Data successfully loaded from cached files.')
    except OSError:
        print('Unable to load cached data set. Loading from original files...')
        # Carrega todas as noticias do data set (21417 true + 23537 fake = 44954 total)
        if n_news == 'all':
            true = pd.read_csv('./assets/True.csv')
            false = pd.read_csv('./assets/Fake.csv')
        # Carrega n_news noticias de cada tipo, totalizando 2*n_news noticias
        else:
            slice_data_frame(n_news,shuffle)
            true = pd.read_csv(f'./assets/cache_csv/True_{n_news}.csv')
            false = pd.read_csv(f'./assets/cache_csv/Fake_{n_news}.csv')
        print('Data successfully loaded from original files.')

        global lemmatizer
        global stop

        lemmatizer = WordNetLemmatizer()
        stop = set(stopwords.words('english'))

        true['category'] = 1
        false['category'] = -1

        df = pd.concat([true, false])

        Y = df['category']
        Y = [Y.to_numpy()]
        Y = np.transpose(Y)

        print('Title')
        print(df.title.count())
        print('Subject')
        print(df.subject.value_counts())

        df['text'] = df['text'] + " " + df['title']

        del df['title']
        del df['subject']
        del df['date']
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        punctuation = list(string.punctuation)
        stop.update(punctuation)

        print(f'Performing denoise_text...')
        df['text'] = df['text'].apply(denoise_text)

        print(f'Performing TF-IDF vectorization...')
        vectorizer = TfidfVectorizer(
            strip_accents = 'unicode',
            lowercase = True,
            min_df = min_df,
            smooth_idf = True
        )
        X = vectorizer.fit_transform(df['text']).toarray()
        X = StandardScaler().fit_transform(X)
        print(f"Transformed dataframe shape = {X.shape}")

        # print(f'Performing principal component analysis...')
        # X = principal_component_analysis(X)

        # cache data set to filesystem (numpy file format)
        np.savez_compressed(f'./assets/cache_npz/X_{n_news}.npz', X)
        np.savez_compressed(f'./assets/cache_npz/Y_{n_news}.npz', Y)
        print('Data set successfully cached for further reuse.')

    # logging the data read from the filesystem
    print(f'len(X) = {len(X)}, len(X[0]) = {len(X[0])}')
    print(f'len(Y) = {len(Y)}, len(Y[0]) = {len(Y[0])}')

    return X, Y
