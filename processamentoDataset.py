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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

#Doing stemming
def lemma(text):
    for i in text.split():
        word_tokens = word_tokenize(i)
        word_filtered = []
        for w in word_tokens:
            w = lemmatizer.lemmatize(w, pos='v')
            word_filtered.append(w)
    return " ".join(w for w in word_filtered)


#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    text = lemma(text)
    return text

def pre_processamento():
# Utilizando o Dataset reduzido que pode ser gerado no script quebra_df
    true = pd.read_csv("./assets/sliced_true.csv")
    false = pd.read_csv("./assets/sliced_fake.csv")

    global lemmatizer
    global stop

    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
# Utilizando o Dataset original
    # true = pd.read_csv("./assets/True.csv")
    # false = pd.read_csv("./assets/Fake.csv")

    nltk.download('stopwords')


    true['category'] = 1
    false['category'] = -1

    df = pd.concat([true, false])

    result = df['category']
    result = [result.to_numpy()]
    result = np.transpose(result)

    # print(df.isna().sum())
    print('Title')
    print(df.title.count())
    print('Subject')
    print(df.subject.value_counts())

    # Gráfico de quantidade de notícias divididas entre True e Fake
    # sns.set_style("darkgrid")
    # veracityChart = sns.countplot(data=df, x="category")
    # plt.title('Number of news divided in True or Fake')
    # for p in veracityChart.patches:
    #     veracityChart.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    # Gráfico de quantidade de notícias divididas entre seus assuntos
    # plt.figure(figsize = (12,8))
    # sns.set(style = "whitegrid",font_scale = 1.2)
    # subjectChart = sns.countplot(x = "subject", hue = "category" , data = df )
    # plt.title('Number of news divided in subjects')
    # for p in subjectChart.patches:
    #     subjectChart.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    # subjectChart.set_xticklabels(subjectChart.get_xticklabels(),rotation=90)

    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    punctuation = list(string.punctuation)
    stop.update(punctuation)

    #Apply function on review column
    df['text']=df['text'].apply(denoise_text)

    # plt.figure(figsize = (20,20)) # Text that is not Fake
    # wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
    # plt.imshow(wc , interpolation = 'bilinear')
    # plt.title('Most used words in authentic news')


    # plt.figure(figsize = (20,20)) # Text that is not Fake
    # wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == -1].text))
    # plt.imshow(wc , interpolation = 'bilinear')
    # plt.title('Most used words in fake news')

    # plt.show()

    vectorizer = TfidfVectorizer();
    X = vectorizer.fit_transform(df['text'])
    print(X.shape)
    tfidf_vectorizer_news_array = X.toarray()
    tfidf_vectorizer_news_array = np.append(tfidf_vectorizer_news_array, result, axis=1)
    train, test = train_test_split(tfidf_vectorizer_news_array, test_size=0.45, random_state=42)

    result = test[:, - 1]

    train = np.delete(train, np.s_[-1:], axis=1)
    test = np.delete(test, np.s_[-1:], axis=1)

    return train, test, result
