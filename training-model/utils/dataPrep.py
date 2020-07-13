from utils import results, sampling, tfmodel, visualize, wordTokenize

import pandas as pd
import re
import nltk


def getData():
    df_train = pd.read_csv('/train.csv')
    df_train.head()
    print(df_train.head())

    X = df_train[['region_en','category_name_en','user_type','weekend','price','description','description_len','title','title_len','param_1_len','param_2_len','param_3_len','param_1','param_2','param_3']]
    y = df_train[['deal_class_5']]

    X_enc = pd.get_dummies(X, columns=['region_en','user_type','category_name_en'], drop_first = True)
    print(X_enc.head())

    from nltk.corpus import stopwords
    nltk.download("stopwords")

    stopWords = stopwords.words('russian')

    X_enc['description_non_stop'] = X_enc['description'].str.replace(r'\d+','')
    X_enc['description_non_stop'] = X_enc['description_non_stop'].apply(lambda x: ' '.join([re.sub(r'[^\w\s]',' ',word) for word in x.split()]))
    X_enc['description_non_stop'] = X_enc['description_non_stop'].apply(lambda x: ' '.join([word.lower().strip() for word in x.split() if word.lower().strip() not in (stopWords) and len(word)>=3 ]))

    X_enc['title_non_stop'] = X_enc['title'].str.replace(r'\d+','')
    X_enc['title_non_stop'] = X_enc['title_non_stop'].apply(lambda x: ' '.join([re.sub(r'[^\w\s]',' ',word) for word in x.split()]))
    X_enc['title_non_stop'] = X_enc['title_non_stop'].apply(lambda x: ' '.join([word.lower().strip() for word in x.split() if word.lower().strip() not in (stopWords) and len(word)>=3 ]))
    print(X_enc['description_non_stop'].head())

    return X_enc, y
