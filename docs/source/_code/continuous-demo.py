from pypair.association import continuous_continuous
from pypair.continuous import Continuous

x = [x for x in range(10)]
y = [y for y in range(10)]

for m in Continuous.measures():
    r = continuous_continuous(x, y, m)
    print(f'{r}: {m}')

print('-' * 15)

con = Continuous(x, y)
for m in con.measures():
    r = con.get(m)
    print(f'{r}: {m}')
