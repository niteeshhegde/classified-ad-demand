from tensorflow import keras as k1
from keras.layers import Input
from keras.layers import Embedding, concatenate,Dense,Dropout,SpatialDropout1D,Flatten
from keras import Model
from keras.callbacks import ModelCheckpoint

import tensorflow_addons as tfa


def create_model(x_train_title, x_train_desc, x_train, y_train, embedding_matrix_title, embedding_matrix_desc):

    embedding_dim=300
    seq_length_title = 20
    seq_length_desc = 60

    nlp_input_desc = Input(shape=(seq_length_title,), name='nlp_input_desc')
    nlp_input_title = Input(shape=(seq_length_desc,), name='nlp_input_title')

    emb1 = Embedding(input_dim=100000, output_dim=embedding_dim, weights=[embedding_matrix_title])(nlp_input_title)
    emb1 = SpatialDropout1D(0.3)(emb1)
    emb1 = Flatten()(emb1)

    emb2 = Embedding(input_dim=100000, output_dim=embedding_dim, weights=[embedding_matrix_desc])(nlp_input_desc)
    emb2 = SpatialDropout1D(0.3)(emb2)
    emb2 = Flatten()(emb2)

    meta_input = Input(shape=(82,), name='meta_input')

    x = concatenate([emb1,emb2, meta_input])

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.05)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=[nlp_input_desc,nlp_input_title, meta_input], outputs=[x])

    early_stopping = k1.callbacks.EarlyStopping(
        monitor='accuracy',
        verbose=1,
        patience=30,
        mode='max',
        restore_best_weights=True)

    model.compile(optimizer=k1.optimizers.Adam(lr=2e-4),
                  loss="categorical_crossentropy",
                  metrics=[tfa.metrics.F1Score(num_classes=3, average="macro", threshold=None), "accuracy"])

    checkpoint_path = "gs://dataproc-e3bd1f7b-2e29-4da6-a5c4-077c164fd32a-us-central1/avito/dl5-200/fasttext/cp-{epoch:04d}.ckpt"

    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=3)

    history = model.fit([x_train_title, x_train_desc, x_train], y_train, epochs=50, callbacks=[cp_callback, early_stopping], validation_split=0.2, shuffle= True,batch_size=2048)
    return model, history
