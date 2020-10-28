import string
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import nltk
from pylab import *
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
ioff()


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

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

if __name__ == "__main__":
    true = pd.read_csv("./assets/True.csv")
    false = pd.read_csv("./assets/Fake.csv")

    nltk.download('stopwords')

    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true, false])

    # print(df.isna().sum())
    print('Title')
    print(df.title.count())
    print('Subject')
    print(df.subject.value_counts())

    # Gráfico de quantidade de notícias divididas entre True e Fake
    sns.set_style("darkgrid")
    sns.countplot(data=df, x="category")
    plt.show()
    # Gráfico de quantidade de notícias divididas entre seus assuntos
    # plt.figure(figsize = (12,8))
    # sns.set(style = "whitegrid",font_scale = 1.2)
    # chart = sns.countplot(x = "subject", hue = "category" , data = df)
    # chart.set_xticklabels(chart.get_xticklabels(),rotation=90)


    # df['text'] = df['text'] + " " + df['title']
    # del df['title']
    # del df['subject']
    # del df['date']

    # stop = set(stopwords.words('english'))
    # punctuation = list(string.punctuation)
    # stop.update(punctuation)

    # #Apply function on review column
    # df['text']=df['text'].apply(denoise_text)

    # plt.figure(figsize = (20,20)) # Text that is not Fake
    # wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
    # plt.imshow(wc , interpolation = 'bilinear')

    # plt.show()