import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def get_data():
    x, y = read_data('/train.csv',
                     x_cols=['region_en', 'category_name_en', 'user_type', 'weekend', 'price', 'description',
                             'description_len', 'title',
                             'title_len', 'param_1_len', 'param_2_len', 'param_3_len', 'param_1', 'param_2', 'param_3'],
                     y_col=['deal_class_5'])
    x = one_hot_encode_data(x, cols=['region_en', 'user_type', 'category_name_en'])
    stop_words = download_stopwords('russian')
    x = remove_stopwords(x, 'description', 'description_non_stop', stop_words)
    x = remove_stopwords(x, 'title', 'title_non_stop', stop_words)
    return x, y


def read_data(path, x_cols, y_col):
    df_train = pd.read_csv(path)
    x = df_train[x_cols]
    y = df_train[y_col]
    return x, y


def one_hot_encode_data(x, cols):
    return pd.get_dummies(x, columns=cols, drop_first=True)


def download_stopwords(lang):
    nltk.download("stopwords")
    stop_words = stopwords.words(lang)
    return stop_words


def remove_stopwords(x, col, new_col, stop_words):
    x[new_col] = x[col].str.replace(r'\d+', '')
    x[new_col] = x[new_col].apply(lambda x1: ' '.join([re.sub(r'[^\w\s]', ' ', word) for word in x1.split()]))
    x[new_col] = x[new_col].apply(lambda x1: ' '.join(
        [word.lower().strip() for word in x1.split() if word.lower().strip() not in stop_words and len(word) >= 3]))
    return x


def split_data(x, y, test_size=0.20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test
