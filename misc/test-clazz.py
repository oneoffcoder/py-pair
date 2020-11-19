class A(object):
    @property
    def a1(self):
        return 'a1'

    @property
    def a2(self):
        return 'a2'

    @property
    def _a3(self):
        return 'a3'

    @property
    def __a3(self):
        return 'a3'


class B(A):
    @property
    def b1(self):
        return 'b1'

    @property
    def b2(self):
        return 'b2'


class C(B):
    @property
    def c1(self):
        return 'c1'


def get_properties(clazz):
    from itertools import chain

    is_property = lambda v: isinstance(v, property)
    is_public = lambda n: not n.startswith('_')
    is_valid = lambda n, v: is_public(n) and is_property(v)

    return list(chain(*[[n for n, v in vars(c).items() if is_valid(n, v)] for c in clazz.__mro__]))


print([n for n, _ in vars(A).items()])
print([n for n, _ in vars(B).items()])
print([n for n, _ in vars(C).items()])

print(list(A.__mro__))
print(list(B.__mro__))
print(list(C.__mro__))

print('-' * 15)

print(get_properties(C))
