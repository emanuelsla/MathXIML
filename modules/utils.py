import pandas as pd
import numpy as np


def generate_random_df(df: pd.DataFrame, size: int, seed=None) -> pd.DataFrame:
    if seed:
        np.random.seed(seed)
    columns = list(df.columns)
    random_df_dict = {}
    for col in columns:
        pop = list(df[col])
        sample = np.random.choice(pop, size, replace=True)
        random_df_dict[col] = list(sample)
    random_df = pd.DataFrame.from_dict(random_df_dict)
    return random_df
