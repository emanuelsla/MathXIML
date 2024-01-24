import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


nominal_column_dict = {'disease': ['no', 'yes'],
                       'sex': ['female', 'male'],
                       'cp': ['typicalangina', 'atypicalangina', 'nonanginal', 'asymptomatic'],
                       'fbs': ['<120', '>120'],
                       'restecg': ['normal', 'sttabnormality', 'lvhypertrophy'],
                       'exang': ['no', 'yes'],
                       'thal': ['normal', 'fixeddefect', 'reversibledefect']}
char_column_names = list(nominal_column_dict.keys())

metric_column_names = ['age', 'trestbps', 'chol', 'thalach', 'ca']

cols_to_delete = ['oldpeak', 'slope', 'target']


def heart_label_modifier(X: pd.DataFrame, seed=None) -> list:
    if seed:
        np.random.seed(seed)
    eps = np.random.uniform(0, 1.5, len(X))

    fbs = np.asarray([1 if f == 1 else 0 for f in list(X['fbs'])])  # high sugar
    restecg = np.asarray([1 if e != 0 else 0 for e in list(X['restecg'])])  # not normal restecg
    thal = np.asarray([1 if t != 0 else 0 for t in list(X['thal'])])  # not normal heart health
    chol = np.asarray(X['chol'])  # high cholesterol
    chol = chol / np.max(chol)
    thalach = np.asarray(X['thalach'])  # high max heart rate
    thalach = chol / np.max(thalach)

    Z = -2.3 + fbs + restecg + thal + chol - thalach + eps
    Y = 1 / (1 + np.exp(-Z))
    Y = [1 if y > 0.5 else 0 for y in Y]

    return Y


def preprocess_heart_data(path: str, test_ratio: float,
                          custom_mechanism=True, seed=None) -> ((pd.DataFrame, pd.DataFrame), (dict, dict)):
    """
    function to preprocess heart data set

    :param path: path to csv
    :param test_ratio: ratio of test data
    :param custom_mechanism: modifies risk column according specified mechanism
    :param seed: random seed
    :return: ((df_train, df_test), (label encoder dict, scaler dict))
    """

    # load data
    df = pd.read_csv(path, index_col=False)

    # refactor columns
    df['disease'] = df['target']
    df = df.drop(columns=cols_to_delete)
    df['thal'] = [x-1 for x in list(df['thal'])]

    # integrate value names for nominal cols
    for key, val in nominal_column_dict.items():
        df[key] = df[key].apply(lambda x: val[x])

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
        df['disease'] = heart_label_modifier(df.drop(columns=['disease']), seed)

    # train test split
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, metric_transformers
