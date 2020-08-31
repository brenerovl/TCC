import pandas as pd
import nltk
import csv
import wandb
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

wandb.init(project="test-project")

wandb.config.test_size = 0.3
wandb.config.seed = 0

df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
news_content_test = []
result_content_test = []

# Remove palavras que aparecem em mais de 40% das notícias
TfidfVectorizer(df,analyzer='word', max_df=0.4)

# Fazendo Lematização/Stemização , stop words usando NLK e retirando caracteres estranhos/espaços
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

for k, t in enumerate(df['text']):
    if k <= 4:
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        dataset_sent = str(t)
        #print(dataset_sent, '\n')

        print('* Fazendo a lematização da', k ,'ª noticia')
        dataset_sent = lemmatizer.lemmatize(dataset_sent, pos='a')

        print('* Removendo Stop Words e fazendo a stemização da', k ,'ª noticia')
        word_tokens = word_tokenize(dataset_sent)

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(ps.stem(w))
        print('* Retirando caracteres estranhos e espaços \n')
        news_content_test.append(' '.join(c for c in filtered_sentence if c.isalnum()))
        result_content_test.append(df['label'][k])

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

# O índice passa a ser a coluna "Unnamed: 0"
# df = df.set_index("Unnamed: 0")

# Coluna mostrando apenas "REAL" ou "FAKE"
# result = df.label

# Dataset sem a label "REAL" ou "FAKE"
# newDf = df.drop(columns="label")
#newDf = df.drop("label", axis=1)

# X -> notícias y-> resultados(fake or real)
X_train, X_test, y_train, y_test = train_test_split(news_content_test, result_content_test, test_size=wandb.config.test_size, random_state=wandb.config.seed)

new_test = X_test

train_matrix = np.array(X_train).reshape(-1, 1)
test_matrix = new_test


# IsolationForest
# Treino
print('* Treinando modelo com IsolationForest')
isolationModel = IsolationForest(random_state=0).fit(train_matrix) # Verificar random state
# Teste
# Deve ser comparado com o y_test

print('* Testando modelo')
isolationTestResult = isolationModel.predict(test_matrix)

wandb.log({'test': isolationTestResult})
print(isolationTestResult)

wandb.sklearn.plot_class_proportions(y_train, y_test, ['TRUE', 'FAKE'])

# EllipticEnvelope

# LocalOutlierFactor

# OneClassSVM