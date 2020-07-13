import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics


def evaluate_train_data(model, x_train_title, x_train_desc,x_train, y_train, batch_size=4096):
    print('\n# Evaluate on train data')
    train_result = model.evaluate([x_train_title, x_train_desc, x_train], y_train, batch_size)
    print('train loss, train acc:', train_result)


def evaluate_test_data(model, x_test_title, x_test_desc, x_test, y_test, batch_size=1024):
    print('\n# Evaluate on test data')
    test_results = model.evaluate([x_test_title, x_test_desc, x_test], y_test, batch_size)
    print('test loss, test acc:', test_results)


def predict(model, x_test, y_test, x_test_title, x_test_desc):
    actual = np.argmax(y_test, axis=1)
    print('\n# Generate predictions')
    predictions = model.predict([x_test_title, x_test_desc, x_test])
    result = np.argmax(predictions, axis=1)
    metrics.confusion_matrix(actual, result)
    print(classification_report(actual,result))
