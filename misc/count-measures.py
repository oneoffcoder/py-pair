from pypair.biserial import Biserial
from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.continuous import Concordance, CorrelationRatio, Continuous

measures = [
    ('Binary-Binary', BinaryTable.measures()),
    ('Confusion Matrix, Binary-Binary', ConfusionMatrix.measures()),
    ('Categorical-Categorical', CategoricalTable.measures()),
    ('Categorical-Continuous, Biserial', Biserial.measures()),
    ('Categorical-Continuous', CorrelationRatio.measures()),
    ('Ordinal-Ordinal, Concordance', Concordance.measures()),
    ('Continuous-Continuous', Continuous.measures())
]
print(sum([len(m) for _, m in measures]))

for n, items in measures:
    title = f'{n} ({len(items)})'
    print(title)
    print('-' * len(title))
    print('')
    for m in items:
        print(f'- {m}')
    print('')

