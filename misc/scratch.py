from pypair.contingency import BinaryMeasures

bm = BinaryMeasures(20, 10, 10, 30)
measures = bm.measures()
for m in measures:
    r = bm.get(m)
    print(f'{m}: {r}')

print('-' * 15)
print(bm.gk_lambda)
print(bm.gk_lambda_reversed)
print(bm.mutual_information)
print(bm.uncertainty_coefficient)
print(bm.uncertainty_coefficient_reversed)
