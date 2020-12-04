import string
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import nltk
import csv
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import decomposition

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
def principal_component_analysis(X,n):
    pca = decomposition.PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)
    return X

def plots(df):
    # Grafico de quantidade de noticias divididas entre True e Fake
    # sns.set_style("darkgrid")
    # veracityChart = sns.countplot(data=df, x="category")
    # plt.title('Number of news divided in True or Fake')
    # for p in veracityChart.patches:
    #     veracityChart.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    # Grafico de quantidade de noticias divididas entre seus assuntos
    # plt.figure(figsize = (12,8))
    # sns.set(style = "whitegrid",font_scale = 1.2)
    # subjectChart = sns.countplot(x = "subject", hue = "category" , data = df )
    # plt.title('Number of news divided in subjects')
    # for p in subjectChart.patches:
    #     subjectChart.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    # subjectChart.set_xticklabels(subjectChart.get_xticklabels(),rotation=90)
    pass

def wordcloud(df):
    # plt.figure(figsize = (20,20)) # Text that is not Fake
    # wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
    # plt.imshow(wc , interpolation = 'bilinear')
    # plt.title('Most used words in authentic news')

    # plt.figure(figsize = (20,20)) # Text that is not Fake
    # wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == -1].text))
    # plt.imshow(wc , interpolation = 'bilinear')
    # plt.title('Most used words in fake news')
    pass

def pre_processamento():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Utilizando o Dataset reduzido que pode ser gerado no script quebra_df
    true = pd.read_csv("./assets/sliced_true.csv")
    false = pd.read_csv("./assets/sliced_fake.csv")

    # Utilizando o Dataset original
    # true = pd.read_csv("./assets/True.csv")
    # false = pd.read_csv("./assets/Fake.csv")

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

    # debugging the data frame
    # print(df.to_string())
    # print(df.columns.tolist())
    # for i in range(10):
    #     print(f'X[{i} = {X[i]} mean = {np.mean(X[i])} sum = {np.sum(X[i])}')

    # wordcloud()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()

    sum = 0
    for i in range(len(X)):
        sum = sum + len(X[i])
    n = int(np.ceil(np.sqrt(sum/len(0.7*X))))
    X = principal_component_analysis(X,n)

    return X, Y
