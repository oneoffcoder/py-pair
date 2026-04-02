![pypair logo](https://py-pair.readthedocs.io/_images/logo.png)

# PyPair

PyPair computes pairwise association measures between variables (binary, categorical, ordinal, and continuous), with local NumPy-first implementations and Spark dataframe support.

## Modern toolchain (Python 3.13 + uv)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync
uv run pytest
```

Build wheel/sdist:

```bash
uv build
```

## Profiling

Run the built-in profiling workload:

```bash
make profile
```

This writes `cProfile` stats to `.profiles/pypair.prof`, prints the hottest frames, and can also emit
internal timings for the decorated contingency measures. It also writes a memory report with
`tracemalloc` allocation hotspots and process peak RSS to `.profiles/pypair.memory.txt`. Override the
default workload or scale with `PROFILE_FLAGS` when needed:

```bash
make profile PROFILE_FLAGS="--workload corr --size 8000 --width 20 --limit 40 --output .profiles/corr.prof --memory-output .profiles/corr.memory.txt"
```

## Quick usage

```python
from pypair.association import (
    binary_binary,
    confusion,
    categorical_categorical,
    binary_continuous,
    concordance,
    categorical_continuous,
    continuous_continuous,
)

# same public convenience API
jaccard = binary_binary(a, b, measure='jaccard')
acc = confusion(a, b, measure='acc')
phi = categorical_categorical(a, b, measure='phi')
biserial = binary_continuous(a, b, measure='biserial')
tau = concordance(a, b, measure='kendall_tau')
eta = categorical_continuous(a, b, measure='eta')
pearson = continuous_continuous(a, b, measure='pearson')
```

## DataFrame integration

- **Pandas**: use `pypair.util.corr(df, func)` to build pairwise association matrices.
- **PySpark**: use `pypair.spark.*` methods for distributed pairwise computations.

## Notes on internals

- Internals now prefer **NumPy** for local numeric workflows where possible.
- Pandas remains supported as a dataframe input/output layer.
- PySpark APIs are preserved for distributed dataframe workflows.
