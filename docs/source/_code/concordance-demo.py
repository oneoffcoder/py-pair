from pypair.association import concordance
from pypair.continuous import Concordance

a = [1, 2, 3]
b = [3, 2, 1]

for m in Concordance.measures():
    r = concordance(a, b, m)
    print(f'{r}: {m}')

print('-' * 15)

con = Concordance(a, b)
for m in con.measures():
    r = con.get(m)
    print(f'{r}: {m}')
