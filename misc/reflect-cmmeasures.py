from pypair.contingency import CmMeasures

get_data = lambda x, y, n: [(x, y) for _ in range(n)]
data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
a = [a for a, _ in data]
b = [b for _, b in data]

m = CmMeasures(207, 282, 231, 242)

for p, v in vars(m).items():
    print(f'{p}: {v}')

print('-' * 15)

measures = CmMeasures.get_measures()
for measure in measures:
    try:
        print(f'{measure}: {m.get(measure)}')
    except ValueError as ve:
        print(f'* {measure}: {ve}')

print('=' * 15)
for measure in measures:
    print(f'- ``{measure}``')
