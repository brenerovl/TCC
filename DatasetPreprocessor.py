import csv
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
from utils import sliceDataFrame

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def strip_strange_symbols(text):
    return re.sub(r'[\W_]+', ' ', text)

def to_lower_case(text):
    return text.lower()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

# Doing stemming
def lemma(text):
    word_filtered = []
    for i in text.split():
        word_tokens = word_tokenize(i)
        for w in word_tokens:
            w = lemmatizer.lemmatize(w, pos='v')
            word_filtered.append(w)
    return " ".join(w for w in word_filtered)

#R emoving the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = strip_strange_symbols(text)
    text = to_lower_case(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    text = lemma(text)
    return text

# perform principal component analysis
def principal_component_analysis(X):
    n_pca_components = int(np.floor(np.sqrt(min(len(X),len(X[0])))))
    print(f'n_components = {n_pca_components}')
    pca = decomposition.PCA(n_components=n_pca_components,svd_solver='full')
    pca.fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    X = pca.transform(X)
    return X

def load_and_preprocess(sliceAmount=-1):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Utilizando o Dataset original
    if sliceAmount == -1:
        true = pd.read_csv('./assets/True.csv')
        false = pd.read_csv('./assets/Fake.csv')
    # Utilizando o Dataset reduzido que pode ser gerado no script quebra_df
    else:
        sliceDataFrame(sliceAmount)
        true = pd.read_csv('./assets/True_Sliced.csv')
        false = pd.read_csv('./assets/Fake_Sliced.csv')

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

    # print(df.isna().sum())
    print('Title')
    print(df.title.count())
    print('Subject')
    print(df.subject.value_counts())

    # plots(df)

    df['text'] = df['text'] + " " + df['title']

    del df['title']
    del df['subject']
    del df['date']
    del df['Unnamed: 0']

    punctuation = list(string.punctuation)
    stop.update(punctuation)

    # Apply function on review column
    df['text'] = df['text'].apply(denoise_text)

    # wordcloud()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()
    X = StandardScaler().fit_transform(X)
    print(f'n_samples = {len(X)}, n_features = {len(X[0])}')
    X = principal_component_analysis(X)

    return X, Y
