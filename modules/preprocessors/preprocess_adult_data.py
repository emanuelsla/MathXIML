import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

nominal_column_names = ['occupation', 'race', 'gender', 'maritalstatus', 'workclass', 'income']
metric_column_names = ['educationalnum', 'hoursperweek', 'age']
ordinal_column_names = []
char_column_names = ordinal_column_names + nominal_column_names

ordinal_encoding = {}

y_name = 'income'
target = (y_name, 'higher50k')


def adult_label_modifier(X: pd.DataFrame, seed=None) -> list:
    if seed:
        np.random.seed(seed)
    eps = np.random.uniform(0, 4, len(X))

    workclass = np.asarray([1 if w == 2 else 0 for w in list(X['workclass'])])
    educationalnum = np.asarray(X['educationalnum'])
    educationalnum = educationalnum / np.max(educationalnum)
    hoursperweek = np.asarray(X['hoursperweek'])
    hoursperweek = hoursperweek / np.max(hoursperweek)

    Z = -4.25 + workclass + 4 * educationalnum + 5 * hoursperweek + eps
    Y = 1 / (1 + np.exp(-Z))
    Y = [1 if y > 0.5 else 0 for y in Y]

    return Y


def preprocess_adult_data(path: str, test_ratio: float,
                          custom_mechanism=True, seed=None) -> ((pd.DataFrame, pd.DataFrame), (dict, dict)):
    """
    function to preprocess german credit risk data set

    :param path: path to csv
    :param test_ratio: ratio of test data
    :param custom_mechanism: modifies risk column according specified mechanism
    :param seed: random seed
    :return: ((df_train, df_test), (label encoder dict, scaler dict))
    """

    # load data
    df = pd.read_csv(path)

    df = df.rename(columns=lambda x: x.lower().replace('-', ''))
    df = df.replace('?', np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)

    drop_cols = ['fnlwgt', 'capitalgain', 'capitalloss', 'education', 'relationship', 'nativecountry']
    for col in drop_cols:
        df = df.drop(col, axis=1)

    for col in nominal_column_names:
        df[col] = df[col].apply(lambda x: str(x).replace('-', '').lower())
        df[col] = df[col].apply(lambda x: str(x).replace('&', '').lower())

    # transform character columns into integers
    # save transformer for inverse_transform
    label_transformers = {}
    for col in char_column_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_transformers[col] = le

    # standard scaling of metric columns
    # save transformers for inverse_transform
    metric_transformers = {}
    for col in metric_column_names:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(np.asarray(df[col]).reshape((-1, 1))).flatten()
        metric_transformers[col] = scaler

    if custom_mechanism:
        df['income'] = adult_label_modifier(df.drop(columns=['income']), seed)

    # train test split
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, metric_transformers
