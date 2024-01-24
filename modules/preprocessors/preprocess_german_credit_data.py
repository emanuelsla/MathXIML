import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ordinal_column_names = ['savingaccounts', 'checkingaccount', 'job']
nominal_column_names = ['sex', 'housing', 'purpose', 'risk']
metric_column_names = ['age', 'duration']
y_name = 'risk'
target = (y_name, 'good')

ordinal_encoding = {'checkingaccount': ['little', 'moderate', 'rich'],
                    'savingaccounts': ['little', 'moderate', 'quiterich', 'rich'],
                    'job': ['unskilled_no_res', 'unskilled_res', 'skilled', 'highlyskilled']}

char_column_names = ordinal_column_names + nominal_column_names


def german_credit_label_modifier(X: pd.DataFrame, seed=None) -> list:
    if seed:
        np.random.seed(seed)
    eps = np.random.uniform(0, 1, len(X))

    housing = np.asarray([1 if h == 1 else 0 for h in list(X['housing'])])
    job = np.asarray([1 if j <= 1 else 0 for j in list(X['job'])])
    age = np.asarray([a if a <= 60 else 0 for a in list(X['age'])])
    age = age / 60

    Z = -1 + 0.5 * job + X['savingaccounts'] + X['checkingaccount'] + age/10 + 0.5 * housing + eps
    Y = 1 / (1 + np.exp(-Z))
    Y = [1 if y > 0.7 else 0 for y in Y]

    return Y


def preprocess_german_credit_data(path: str, test_ratio: float,
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
    df = pd.read_csv(path, index_col=0)

    # remove whitespaces from column and value names
    df = df.rename(columns=lambda x: x.lower().replace(' ', ''))
    df['savingaccounts'] = df['savingaccounts'].apply(lambda x: str(x).replace(' ', ''))
    df['purpose'] = df['purpose'].apply(lambda x: str(x).replace(' ', ''))
    df = df.drop('creditamount', axis=1)
    df = df.replace('nan', np.nan, regex=True)
    df = df.dropna()

    # transform job into ordinal value
    job_dict = {'0': 'unskilled_no_res', '1': 'unskilled_res', '2': 'skilled', '3': 'highlyskilled'}
    df['job'] = df['job'].apply(lambda x: job_dict[str(x)])

    # transform character columns into integers
    # save transformer for inverse_transform
    label_transformers = {}
    for col in char_column_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_transformers[col] = le

    if custom_mechanism:
        df['risk'] = german_credit_label_modifier(df.drop(columns=['risk']), seed)

    # standard scaling of metric columns
    # save transformers for inverse_transform
    metric_transformers = {}
    for col in metric_column_names:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(np.asarray(df[col]).reshape((-1, 1))).flatten()
        metric_transformers[col] = scaler

    # train, test, explanation data split
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, metric_transformers
