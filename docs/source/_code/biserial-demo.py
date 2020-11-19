from pypair.association import binary_continuous
from pypair.biserial import Biserial

get_data = lambda x, y, n: [(x, y) for _ in range(n)]
data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
a = [a for a, _ in data]
b = [b for _, b in data]

for m in Biserial.measures():
    r = binary_continuous(a, b, m)
    print(f'{r}: {m}')

print('-' * 15)

biserial = Biserial(a, b)
for m in biserial.measures():
    r = biserial.get(m)
    print(f'{r}: {m}')
