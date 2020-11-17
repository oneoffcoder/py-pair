from pypair.association import binary_binary
from pypair.contigency import BinaryTable

get_data = lambda x, y, n: [(x, y) for _ in range(n)]
data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
a = [a for a, _ in data]
b = [b for _, b in data]

for m in BinaryTable.measures():
    r = binary_binary(a, b, m)
    print(f'{r}: {m}')

print('-' * 15)

table = BinaryTable(a, b)
for m in table.measures():
    r = table.get(m)
    print(f'{r}: {m}')