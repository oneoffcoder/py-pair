from pypair.association import categorical_continuous
from pypair.continuous import CorrelationRatio

data = [
        ('a', 45), ('a', 70), ('a', 29), ('a', 15), ('a', 21),
        ('g', 40), ('g', 20), ('g', 30), ('g', 42),
        ('s', 65), ('s', 95), ('s', 80), ('s', 70), ('s', 85), ('s', 73)
    ]
x = [x for x, _ in data]
y = [y for _, y in data]
for m in CorrelationRatio.measures():
    r = categorical_continuous(x, y, m)
    print(f'{r}: {m}')

print('-' * 15)

cr = CorrelationRatio(x, y)
for m in cr.measures():
    r = cr.get(m)
    print(f'{r}: {m}')
