![pypair logo](https://py-pair.readthedocs.io/_images/logo.png)

# PyPair

PyPair is a statistical library to compute pairwise association between any two variables.

- [Documentation](https://py-pair.readthedocs.io/)
- [PyPi](https://pypi.org/project/pypair/) 
- [Gitter](https://gitter.im/dataflava/py-pair)

Here's a short and sweet snippet for using the API against a dataframe that stores strictly binary data.

```python
from pypair.association import binary_binary

jaccard = lambda a, b: binary_binary(a, b, measure='jaccard')
tanimoto = lambda a, b: binary_binary(a, b, measure='tanimoto_i')

df = get_a_pandas_binary_dataframe()

jaccard_corr = df.corr(method=jaccard)
tanimoto_corr = df.corr(method=tanimoto)

print(jaccard_corr)
print('-' * 15)
print(tanimoto_corr)
```

# Software Copyright

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

# Book Copyright

Copyright 2020 One-Off Coder

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) by [One-Off Coder](https://www.oneoffcoder.com).

![Creative Commons Attribution 4.0 International License](https://i.creativecommons.org/l/by/4.0/88x31.png "Creative Commons Attribution 4.0 International License")

# Art Copyright

Copyright 2020 Daytchia Vang

# Citation

```
@misc{oneoffcoder_pypair_2020,
title={PyPair, A Statistical API for Bivariate Association Measures},
url={https://github.com/oneoffcoder/py-pair},
author={Jee Vang},
year={2020},
month={Nov}}
```
