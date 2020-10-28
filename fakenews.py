import pandas as pd
import nltk
import csv
import math
# import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# wandb.init(project="test-project")

# wandb.config.test_size = 0.3
# wandb.config.seed = 0
test_size = 1
seed = 53

df = pd.read_csv("./assets/fake_or_real_news.csv")
news_content_array = []
news_content_string = []
result_content_test = []
news_tf = []
news_idf = {}
news_tf_idf = []

# Fazendo Lematização , stop words usando NLK e retirando caracteres estranhos/espaços
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

word_map = {}
num_words = 0
tf_adjusted = {}

import json

def escrever_json(conteudo, nome_arquivo):
    with open(nome_arquivo, 'w') as f:
        json.dump(conteudo, f)

def word_map_couter(word_list):
    for w in word_list:
        if w in word_map:
            word_map[w] = word_map[w] + 1
        else:
            word_map[w] = 1

def count_words(word_map):
    count = 0
    for w in word_map:
        count = word_map[w] + count

    return count
# TF(t) = (quantidade de vezes que o termo t aparece no documento) / (número total de termos do documento)
# Calcula o quão frequente é a ocorrencia de um termo específico em um documento.

# IDF(t) = log_e((número total de documentos) / (número de documentos que contém o termo t))
# Calcula a importancia do termo t

# TF_IDF(t) = TF(t) * IDF(t).

def term_frequency(termo, documento):
    frequencia_t = 0
    for t in documento:
        if t == termo:
            frequencia_t = frequencia_t + 1
    
    return frequencia_t / len(documento)

def inverse_document_frequency(termo, documentos):
    qtd_d_with_termo = 0
    for d in documentos:
        if (termo in d):
            qtd_d_with_termo = qtd_d_with_termo + 1

    return math.log(len(documentos) / qtd_d_with_termo)

def calc_idf():
    for d in news_content_array:
        for t in d:
            if t not in news_idf:
                news_idf[t] = inverse_document_frequency(t, news_content_array)

def calc_tf_idf():
    for d_tf in news_tf:
        tfidf = {}
        for w_tf in d_tf:
            if w_tf not in tfidf:
                tfidf[w_tf] = d_tf[w_tf] * news_idf[w_tf]
        news_tf_idf.append(tfidf)

for k, t in enumerate(df['text']):
    if k < 4:
        print('* Fazendo a lematização e retirando caracteres da', k ,'ª noticia')
        word_tokens = word_tokenize(t)
        word_filtered = []
        for w in word_tokens:
            w = ''.join(c for c in w if c.isalnum()).lower()
            w = lemmatizer.lemmatize(w, pos='v')
            word_filtered.append(w)

        filtered_sentence = ' '.join(w for w in word_filtered)
        
        print('* Removendo Stop Words da', k + 1 ,'ª noticia')
        word_tokens = word_tokenize(filtered_sentence)
        word_tokens_filtered = [w for w in word_tokens if not w in stop_words]
        sentence_not_stop_word = ' '.join(w for w in word_tokens if not w in stop_words)
        doc_tf = {}
        for t in word_tokens_filtered:
            if t not in doc_tf:
                doc_tf[t] = term_frequency(t, word_tokens_filtered)

        word_map_couter(word_tokens_filtered)
        print('* Retirando caracteres estranhos e espaços \n')
        news_content_array.append(word_tokens_filtered)
        news_content_string.append(sentence_not_stop_word)
        news_tf.append(doc_tf)
        print(news_content_array)
        result_content_test.append(df['label'][k])

calc_idf()
calc_tf_idf()
escrever_json(news_tf, 'tf.json')
escrever_json(news_idf, 'idf.json')
escrever_json(news_tf_idf, 'tf_idf.json')
num_words = count_words(word_map)
print('Número de palavras no documento: ', num_words)
sortedDict = sorted(word_map.items(), key=lambda x: x[1], reverse=True)
print('Número de palavras diferentes: ', len(sortedDict))
print('Palavra que mais apareceu ', sortedDict[0])
# print(df['text'])

# Filtro 2 (Feito acima se quiser pode remover)
# Remove os english stop words
# count_vectorizer = CountVectorizer(stop_words='english')

# Filtro 3 (Feito acima se quiser pode remover)
# Usando max_df
# Remove palavras que aparecem em mais de 40% das notícias
# tfidf_vectorizer = TfidfVectorizer(max_df=0.4)
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer.transform(X_test)

# X -> notícias y-> resultados(fake or real)
#X_train, X_test, y_train, y_test = train_test_split(news_content_array, result_content_test, test_size=test_size, random_state=seed)

# CountVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer_news = count_vectorizer.fit_transform(news_content_string)
count_vectorizer_news_array = count_vectorizer_news.toarray()

# TFIDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer_news = tfidf_vectorizer.fit_transform(news_content_string)
tfidf_vectorizer_news_array = tfidf_vectorizer_news.toarray()

# new_test = count_vectorizer.transform(X_test)

# new_test = X_test

# train_matrix = np.array(X_train).reshape(-1, 1).astype(np.float64)
# test_matrix = new_test

# IsolationForest
# Treino
# print('* Treinando modelo com IsolationForest')
# isolationModel = IsolationForest(random_state=0).fit(train_matrix)

# Teste
# Deve ser comparado com o y_test

# print('* Testando modelo')
# isolationTestResult = isolationModel.predict(test_matrix)

# wandb.log({'test': isolationTestResult})

# wandb.sklearn.plot_class_proportions(y_train, y_test, ['TRUE', 'FAKE'])