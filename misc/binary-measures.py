import pandas as pd

df = pd.read_csv('binary-measures.csv')
print(df)
print(df.columns)
print(df.shape)
print(len(df.equation.unique()))

equations = [r.equation for _, r in df.iterrows()]
print(len(equations))
m = {e: 0 for e in equations}
for e in equations:
    m[e] += 1
for e, c in m.items():
    if c > 1:
        print(e, c)

for _, r in df[df.type == 'd'].sort_values(['name']).iterrows():
    e_name = r['name']
    e_form = r.equation
    name = f'   * - {e_name}'
    eqn = f'     - :math:`{e_form}`'
    print(name)
    print(eqn)
