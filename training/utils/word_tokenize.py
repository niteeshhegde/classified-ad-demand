import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import text, sequence


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_embedding_indexes(path = '/wiki.ru.vec'):
    return dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(path))


def get_tokenizer(max_features=100000, words=[]):
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(words)
    return tokenizer


def get_padded_sequence(tokenizer,x_train,max_len):
    return sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len)


def get_embedding_data(embeddings_index, tokenizer, max_features, embed_size):
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix, embeddings_index


def tokenize(x_train, x_test, y_train, y_test, x_enc, max_features=100000, max_len_title=20, max_len_desc=60, embed_size=300):
    tokenizer_title = get_tokenizer(max_features=max_features, words=list(x_train['title_non_stop'].values)+list(x_test['title_non_stop'].values))
    x_train_title = get_padded_sequence(tokenizer_title, x_train['title_non_stop'].values, max_len_title)
    x_test_title = get_padded_sequence(tokenizer_title, x_test['title_non_stop'].values, max_len_title)
    embeddings_index = get_embedding_indexes()
    embeddings_matrix_title, embeddings_index_title = get_embedding_data(embeddings_index, tokenizer_title, max_features, embed_size)

    tokenizer_desc = get_tokenizer(max_features=max_features, words=list(x_train['description_non_stop'].values)+list(x_test['description_non_stop'].values))
    x_train_desc = get_padded_sequence(tokenizer_desc, x_train['description_non_stop'].values, max_len_desc)
    x_test_desc = get_padded_sequence(tokenizer_desc, x_test['description_non_stop'].values, max_len_desc)
    embeddings_index = get_embedding_indexes()
    embeddings_matrix_desc, embeddings_index_desc = get_embedding_data(embeddings_index, tokenizer_title, max_features, embed_size)

    x_enc = x_enc.drop(columns=['description', 'title', 'description_non_stop', 'title_non_stop', 'param_1', 'param_2', 'param_3'])
    x_train = x_train.drop(columns=['description', 'title', 'description_non_stop', 'title_non_stop', 'param_1', 'param_2', 'param_3'])
    x_test = x_test.drop(columns=['description', 'title', 'description_non_stop', 'title_non_stop', 'param_1', 'param_2',' param_3'])

    sc = StandardScaler()
    x_train.loc[:, ["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]] = sc.fit_transform(x_train[["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]])
    x_test.loc[:, ["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]] = sc.transform(x_test[["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]])

    x_test = x_test.to_numpy()
    y_test = pd.get_dummies(y_test, columns=['deal_class_5']).to_numpy()
    y_train = pd.get_dummies(y_train, columns=['0']).to_numpy()

    return x_train, y_train, x_test, y_test, x_train_title, x_train_desc, x_test_title, x_test_desc, embeddings_matrix_title, embeddings_matrix_desc
