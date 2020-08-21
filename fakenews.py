import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")

# O índice passa a ser a coluna "Unnamed: 0"
df = df.set_index("Unnamed: 0")

# Coluna mostrando apenas "REAL" ou "FAKE"
result = df.label

# Dataset sem a label "REAL" ou "FAKE"
newDf = df.drop(columns="label")
#newDf = df.drop("label", axis=1)

print('Retirando caracteres estranhos')
# Retirando caracteres estranhos e espaços
for k, t in enumerate(df['text']):
    df['text'][k] = ''.join((c for c in t if c.isalnum())).lower()

print(df['text'])

# X -> notícias y-> resultados(fake or real)
X_train, X_test, y_train, y_test = train_test_split(df['text'], result, test_size=0.30, random_state=53)

# Filtro 1
# Remove os english stop words
count_vectorizer = CountVectorizer(stop_words='english')
new_train = count_vectorizer.fit_transform(X_train)
new_test = count_vectorizer.transform(X_test)

# Filtro 2
# Usando max_df
# Remove palavras que aparecem em mais de 40% das notícias
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.4)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# train_matrix = new_train.toarray()
# test_matrix = new_test.toarray()

# # IsolationForest
# # Treino
# isolationModel = IsolationForest(random_state=0).fit(train_matrix) # Verificar random state
# # Teste
# # Deve ser comparado com o y_test
# isolationTestResult = isolationModel.predict(test_matrix)
# print(isolationTestResult)

# EllipticEnvelope

# LocalOutlierFactor

# OneClassSVM