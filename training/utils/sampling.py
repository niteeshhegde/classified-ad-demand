import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder


def resample_data(x_train, x_test, y_train, y_test,  n_samples=700000, y_col='deal_class_5'):

    LE = LabelEncoder()
    y_train[y_col] = LE.fit_transform(y_train.deal_class_5)
    y_test[y_col] = LE.fit_transform(y_test.deal_class_5)

    x_train[y_col] = y_train[y_col]

    df_2 = x_train[x_train[y_col] == 2]
    df_1 = x_train[x_train[y_col] == 1]
    df_0 = x_train[x_train[y_col] == 0]

    # Downsample majority class
    df_2 = resample(df_2,
                    replace=False,  # sample without replacement
                    n_samples=n_samples,  # to match minority class
                    random_state=123)  # reproducible results
    df_1 = resample(df_1,
                    replace=True,  # sample without replacement
                    n_samples=n_samples,  # to match minority class
                    random_state=123)
    df_0 = resample(df_0,
                    replace=True,  # sample without replacement
                    n_samples=n_samples,  # to match minority class
                    random_state=123)

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_2, df_1, df_0])

    y_train = df_downsampled[y_col]
    X_train = df_downsampled.drop(columns=[y_col])

    return X_train, y_train, x_test, y_test

