import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y_name = 'diabetes'
char_column_names = [y_name]
metric_column_names = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi',
                       'diabetespedigreefunction', 'age']


def diabetes_label_modifier(X: pd.DataFrame, seed=None) -> list:
    if seed:
        np.random.seed(seed)
    eps = np.random.uniform(0, 3, len(X))

    bloodpressure = np.asarray(X['bloodpressure'])
    bloodpressure = bloodpressure / np.max(bloodpressure)
    bmi = np.asarray(X['bmi'])
    bmi = bmi / np.max(bmi)
    glucose = np.asarray(X['glucose'])
    glucose = bmi / np.max(glucose)

    Z = -2.25 + glucose + 3 * bmi + 3 * bloodpressure + eps
    Y = 1 / (1 + np.exp(-Z))
    Y = [1 if y > 0.5 else 0 for y in Y]

    return Y


def preprocess_diabetes_data(path: str, test_ratio: float,
                             custom_mechanism=True, seed=None) -> ((pd.DataFrame, pd.DataFrame), (dict, dict)):
    """
    function to preprocess diabetes data set

    :param path: path to csv
    :param test_ratio: ratio of test data
    :param custom_mechanism: modifies risk column according specified mechanism
    :param seed: random seed
    :return: ((df_train, df_test), (label encoder dict, scaler dict))
    """

    # load data
    df = pd.read_csv(path, index_col=False)

    # refactor columns
    df = df.rename(columns=lambda x: x.lower())
    df['diabetes'] = ['yes' if y == 1 else 'no' for y in list(df['outcome'])]
    df = df.drop(columns=['outcome'])

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
        df['diabetes'] = diabetes_label_modifier(df.drop(columns=['diabetes']), seed)

    # train test split
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, metric_transformers
