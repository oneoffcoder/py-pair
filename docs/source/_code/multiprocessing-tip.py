import pandas as pd
import numpy as np
import random
from random import randint
from pypair.association import binary_binary
from itertools import combinations
from multiprocessing import Pool

np.random.seed(37)
random.seed(37)

def get_data(n_rows=1000, n_cols=5):
    data = [tuple([randint(0, 1) for _ in range(n_cols)]) for _ in range(n_rows)]
    cols = [f'x{i}' for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)

def compute(a, b, df):
    x = df[a]
    y = df[b]
    return f'{a}_{b}', binary_binary(x, y, measure='jaccard')

if __name__ == '__main__':
    df = get_data()

    with Pool(10) as pool:
        pairs = ((a, b, df) for a, b in combinations(df.columns, 2))
        bc = pool.starmap(compute, pairs)
    
    bc = sorted(bc, key=lambda tup: tup[0])
    print(dict(bc))