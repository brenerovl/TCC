import pandas as pd

# r_news - Quantidade de notícias que estarão no dataframe reduzido

r_news = 200
n_news = list(range(r_news))

fake_csv = pd.read_csv('./assets/Fake.csv')
true_csv = pd.read_csv('./assets/True.csv')

fake_csv.loc[n_news].to_csv(r'./assets/sliced_fake.csv')
true_csv.loc[n_news].to_csv(r'./assets/sliced_true.csv')
