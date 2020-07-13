import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def resample(x_enc, y):
    X_train, X_test, y_train, y_test = train_test_split(x_enc, y, test_size = 0.20, random_state = 42, stratify=y)

    LE = LabelEncoder()
    y_train['deal_class_5'] = LE.fit_transform(y_train.deal_class_5)
    y_test['deal_class_5'] = LE.fit_transform(y_test.deal_class_5)

    X_train['deal_class_5'] = y_train['deal_class_5']

    df_2 = X_train[X_train['deal_class_5'] == 2]
    df_1 = X_train[X_train['deal_class_5'] == 1]
    df_0 = X_train[X_train['deal_class_5'] == 0]
    # Downsample majority class
    df_2 = resample(df_2,
                    replace=False,  # sample without replacement
                    n_samples=700000,  # to match minority class
                    random_state=123)  # reproducible results
    df_1 = resample(df_1,
                    replace=True,  # sample without replacement
                    n_samples=700000,  # to match minority class
                    random_state=123)
    df_0 = resample(df_0,
                    replace=True,  # sample without replacement
                    n_samples=700000,  # to match minority class
                    random_state=123)
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_2, df_1, df_0])

    # Display new class counts
    print(df_downsampled.deal_class_5.value_counts())

    y_train = df_downsampled['deal_class_5']
    X_train = df_downsampled.drop(columns=['deal_class_5'])

    return X_train, y_train, X_test, y_test

