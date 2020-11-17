# PyPair

PyPair is a statistical library to compute pairwise association between any two variables. In general, statistical variable types are viewed as `categorical` or `continuous`. Categorical variables have no inherit order to their values, while continuous variables do. This API has `over 130 association measures` implemented for any combination of categorical and/or continuous variables. 

The quickest way to get started with PyPair is to use the `convenience` methods whose names indicate the variable pair types.

```python
from pypair.association import binary_binary, categorical_categorical, \
    binary_continuous, concordance, categorical_continuous, continuous_continuous, confusion, agreement

# assume a and b are the appropriate iterables of values for 2 variables
jaccard = binary_binary(a, b, measure='jaccard')
acc = confusion(a, b, measure='acc')
phi = categorical_categorical(a, b, measure='phi')
kappa = agreement(a, b, measure='cohen_k')
biserial = binary_continuous(a, b, measure='biserial')
tau = concordance(a, b, measure='kendall_tau')
eta = categorical_continuous(a, b, measure='eta')
pearson = continuous_continuous(a, b, measure='pearson')
```

# More Information

- [Documentation](https://py-pair.readthedocs.io/)
- [Source Code](https://www.github.com/oneoffcoder/py-pair) 

# Copyright

```
Copyright 2020 One-Off Coder

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# Citation

```
@misc{oneoffcoder_pypair_2020,
title={PyPair, A Statistical API for Bivariate Association Measures},
url={https://github.com/oneoffcoder/py-pair},
author={Jee Vang},
year={2020},
month={Nov}}
```
