from pypair.decorator import similarity
from pypair.table import BinaryTable

a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

t = BinaryTable(a, b)

for p, v in vars(t).items():
    print(f'{p}: {v}')

print('-' * 15)

is_property = lambda v: isinstance(v, property)
is_method = lambda n: not n.startswith('_BinaryTable')
is_valid = lambda n, v: is_property(v) and is_method(n)

measures = [n for n, v in vars(BinaryTable).items() if is_valid(n, v)]
for measure in measures:
    try:
        print(f'{measure}: {getattr(t, measure)}')
    except ValueError as ve:
        print(f'* {measure}: {ve}')
