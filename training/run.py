from utils import data_prep, results, sampling, tfmodel, visualize, word_tokenize


if __name__ == '__main__':
    # Get clean capstone data with stopwords removed from title and description and split the data
    x, y = data_prep.get_data()
    x_full, y_full = x, y
    x_train, x_test, y_train, y_test = data_prep.split_data(x, y)

    # Sample the data for 700000 samples for each class
    X_train, y_train, X_test, y_test = sampling.resample_data(x_train, x_test, y_train, y_test)

    # Get word2vec embeddings in Russian for title and description
    x_train, y_train, x_test, y_test, x_train_title, x_train_desc, x_test_title, x_test_desc, embedding_matrix_title, embedding_matrix_desc = word_tokenize.tokenize(x_train, x_test, y_train, y_test, x_full)

    # Get the tf model
    model, history = tfmodel.create_model(x_test_title, x_test_desc, x_train, y_train, embedding_matrix_title, embedding_matrix_desc)

    # Print model Summary
    print(model.summary())

    # Get Results
    results.evaluate_train_data(model, x_train_title, x_train_desc, x_train, y_train)
    results.evaluate_test_data(model, x_test_title, x_test_desc, x_test, y_test)
    results.predict(model, X_test, y_test, x_test_title, x_test_desc)

    # Get Visual Results
    visualize.visualize(history)
