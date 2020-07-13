from utils import dataPrep, results, sampling, tfmodel, visualize, wordTokenize

# Get clean capstone data with stopwords removed from title and description
X_enc, y = dataPrep.getData()

# Sample the data for 700000 samples for each class
X_train, y_train, X_test, y_test = sampling.resample(X_enc, y)

# Get word2vec embeddings in Russian for title and description
X_train, y_train, X_test, y_test, X_train_title, X_train_desc, X_test_title, X_test_desc, embedding_matrix_title, embedding_matrix_desc = wordTokenize.tokenize(X_train, X_test, y_train, y_test, X_enc, text, sequence)

# Get the tf model
model, history = tfmodel.create_model(X_test_title, X_test_desc, X_train, y_train, embedding_matrix_title, embedding_matrix_desc)

# Print model Summary
print(model.summary())

# Get Results
results.evaluate_train_data(model, X_train_title, X_train_desc, X_train, y_train)
results.evaluate_test_data(model, X_test_title, X_test_desc, X_test, y_test)
results.predict(model, X_test, y_test, X_test_title, X_test_desc)

# Get Visual Results
visualize.visualize(history)
