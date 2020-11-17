from abc import ABC
from functools import lru_cache


class MeasureMixin(ABC):
    """
    Measure mixin. Able to get list the functions decorated with `@property` and also
    access such property based on name.
    """

    @classmethod
    def measures(cls):
        """
        Gets a list of all the measures.

        :return: List of all the measures.
        """
        return get_measures(cls.__name__, cls)

    @lru_cache(maxsize=None)
    def get(self, measure):
        """
        Gets the specified measure.

        :param measure: Name of measure.
        :return: Measure.
        """
        return getattr(self, measure)

    @lru_cache(maxsize=None)
    def get_measures(self):
        """
        Gets a list of all the measures.

        :return: List of all the measures.
        """
        return get_measures(self.__name, self.__clazz)

    @property
    def __name(self):
        """
        Gets the clazz name.

        :return: Clazz name.
        """
        return type(self).__name__

    @property
    def __clazz(self):
        """
        Gets the clazz.

        :return: Clazz.
        """
        return self.__class__


def get_measures(name, clazz):
    """
    Gets all the measures of a clazz.

    :param name: Name of clazz with underscore prefix.
    :param clazz: Clazz.
    :return: List of measures.
    """
    is_property = lambda v: isinstance(v, property)
    is_method = lambda n: not n.startswith(f'_{name}')
    is_valid = lambda n, v: is_property(v) and is_method(n)

    measures = sorted([n for n, v in vars(clazz).items() if is_valid(n, v)])

    return measures
