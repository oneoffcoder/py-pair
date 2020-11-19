from pypair.contingency import CategoricalTable, BinaryTable, ConfusionMatrix, AgreementTable

a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

cat = CategoricalTable(a, b)
bin = BinaryTable(a, b)
con = ConfusionMatrix(a, b)
agr = AgreementTable(a, b)

print(cat.measures())
print(CategoricalTable.measures())
print('-' * 15)
print(bin.measures())
print(BinaryTable.measures())
print('-' * 15)
print(con.measures())
print(ConfusionMatrix.measures())
print('-' * 15)
print(agr.measures())
print(AgreementTable.measures())

print('~' * 15)
print('~' * 15)


def print_measures(computer):
    r = {m: computer.get(m) for m in computer.measures()}
    print(r)


print_measures(cat)
print_measures(bin)
print_measures(con)
print_measures(agr)
