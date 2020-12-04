from random import randint

import pandas as pd

from pypair.association import binary_binary


def get_data(n_rows=1000, n_cols=5):
    data = [tuple([randint(0, 1) for _ in range(n_cols)]) for _ in range(n_rows)]
    cols = [f'x{i}' for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


if __name__ == '__main__':
    jaccard = lambda a, b: binary_binary(a, b, measure='jaccard')
    tanimoto = lambda a, b: binary_binary(a, b, measure='tanimoto_i')

    df = get_data()
    jaccard_corr = df.corr(method=jaccard)
    tanimoto_corr = df.corr(method=tanimoto)

    print(jaccard_corr)
    print('-' * 15)
    print(tanimoto_corr)
