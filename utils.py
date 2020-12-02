import pandas as pd
import numpy as np

# r_news - Quantidade de notícias que estarão no dataframe reduzido

def exponentialList(len):
    upper_exp = np.ceil(np.log2(len))-1
    numbers = np.logspace(start=0,stop=upper_exp,num=upper_exp+1,base=2,dtype='int')+1
    numbers[0] = numbers[0]-1
    return numbers

def quebraDF(news):
    r_news = news
    n_news = list(range(r_news))

    fake_csv = pd.read_csv('./assets/Fake.csv')
    true_csv = pd.read_csv('./assets/True.csv')

    fake_csv.loc[n_news].to_csv(r'./assets/sliced_fake.csv')
    true_csv.loc[n_news].to_csv(r'./assets/sliced_true.csv')
