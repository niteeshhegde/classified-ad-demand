import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def tokenize(X_train, X_test, y_train, y_test, X_enc, text, sequence):
    max_features = 100000
    maxlen_title = 20
    maxlen_desc = 60
    embed_size = 300

    X_train_title = X_train['title_non_stop'].values
    X_test_title = X_test['title_non_stop'].values

    tokenizer_title = text.Tokenizer(num_words=max_features)
    tokenizer_title.fit_on_texts(list(X_test_title)+list(X_train_title))

    X_train_title = tokenizer_title.texts_to_sequences(X_train_title)
    X_train_title = sequence.pad_sequences(X_train_title, maxlen=maxlen_title)


    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('/wiki.ru.vec'))
    embeddings_index_2 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('/wiki.ru.vec'))

    word_index_title = tokenizer_title.word_index
    nb_words = min(max_features, len(word_index_title))
    embedding_matrix_title = np.zeros((nb_words, embed_size))
    for word, i in word_index_title.items():
        if i >= max_features: continue
        embedding_vector_title = embeddings_index.get(word)
        if embedding_vector_title is not None: embedding_matrix_title[i] = embedding_vector_title

    X_train_desc = X_train['description_non_stop'].values
    X_test_desc = X_test['description_non_stop'].values
    tokenizer_desc = text.Tokenizer(num_words=max_features)
    tokenizer_desc.fit_on_texts(list(X_test_desc)+list(X_train_desc))

    X_train_desc = tokenizer_desc.texts_to_sequences(X_train_desc)
    X_train_desc = sequence.pad_sequences(X_train_desc, maxlen=maxlen_desc)

    word_index_desc = tokenizer_desc.word_index
    nb_words = min(max_features, len(word_index_desc))
    embedding_matrix_desc = np.zeros((nb_words, embed_size))
    for word, i in word_index_desc.items():
        if i >= max_features: continue
        embedding_vector_desc = embeddings_index_2.get(word)
        if embedding_vector_desc is not None: embedding_matrix_desc[i] = embedding_vector_desc

    X_enc = X_enc.drop(columns=['description', 'title','description_non_stop', 'title_non_stop','param_1', 'param_2','param_3'])
    X_train = X_train.drop(columns=['description', 'title', 'description_non_stop', 'title_non_stop','param_1', 'param_2','param_3'])
    X_test = X_test.drop(columns=['description', 'title','description_non_stop', 'title_non_stop','param_1', 'param_2','param_3'])

    sc = StandardScaler()
    X_train.loc[:, ["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]] = sc.fit_transform(X_train[["price","description_len","title_len","param_1_len","param_2_len","param_3_len"]])
    X_test.loc[:, ["price", "description_len", "title_len", "param_1_len", "param_2_len", "param_3_len"]] = sc.transform(X_test[["price","description_len","title_len","param_1_len","param_2_len","param_3_len"]])

    X_test = X_test.to_numpy()
    y_test = pd.get_dummies(y_test, columns=['deal_class_5']).to_numpy()
    y_train = pd.get_dummies(y_train, columns=['0']).to_numpy()

    return X_train, y_train, X_test, y_test, X_train_title, X_train_desc, X_test_title, X_test_desc, embedding_matrix_title, embedding_matrix_desc
