import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


nominal_column_dict = {'target': {'e': 'edible', 'p': 'poisonous'},
                       'capshape': {'b': 'bell', 'c': 'conical',
                                    'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
                       'capsurface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
                       'capcolor': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray',
                                    'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
                       'bruises': {'t': 'yes', 'f': 'no'},
                       'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy',
                                'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'},
                       'gillattachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
                       'gillspacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'},
                       'gillsize': {'b': 'broad', 'n': 'narrow'},
                       'gillcolor': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate',
                                     'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple',
                                     'e': 'red', 'w': 'white', 'y': 'yellow'},
                       'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous',
                                      's': 'scattered', 'v': 'several',  'y': 'several'},
                       'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows',
                                   'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
                       }

metric_column_names = []

char_column_names = list(nominal_column_dict.keys())

drop_cols = ['class', 'stalkshape', 'stalkroot', 'stalksurfaceabovering', 'stalksurfacebelowring',
             'stalkcolorabovering', 'stalkcolorbelowring', 'veiltype', 'veilcolor',
             'ringnumber', 'ringtype', 'sporeprintcolor']


def mushrooms_label_modifier(X: pd.DataFrame, seed=None) -> list:
    if seed:
        np.random.seed(seed)
    Y = [0] * len(X)
    poison_colors = ['pink', 'purple', 'red', 'yellow']
    poison_odors = ['fishy', 'foul', 'musty']
    for i in range(len(X)):
        if (X.loc[i, 'odor'] in poison_odors) \
                or (X.loc[i, 'gillcolor'] in poison_colors):
            Y[i] = 1
    return Y


def preprocess_mushrooms_data(path: str, test_ratio: float,
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
    df = df.rename(columns=lambda x: x.lower().replace('-', ''))
    df['target'] = df['class']
    df = df.drop(columns=drop_cols)

    # integrate value names for nominal cols
    for key, val in nominal_column_dict.items():
        df[key] = df[key].apply(lambda x: val[x])

    if custom_mechanism:
        df['target'] = mushrooms_label_modifier(df.drop(columns=['target']), seed)

    # transform character columns into integers
    # save transformer for inverse_transform
    label_transformers = {}
    for col in char_column_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_transformers[col] = le

    # train test split
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, {}
