def get_measures(name, typez):
    """
    Gets all the measures of a clazz.

    :param name: Name of clazz with underscore prefix.
    :param typez: Clazz.
    :return: List of measures.
    """
    is_property = lambda v: isinstance(v, property)
    is_method = lambda n: not n.startswith(name)
    is_valid = lambda n, v: is_property(v) and is_method(n)
    measures = sorted([n for n, v in vars(typez).items() if is_valid(n, v)])
    return measures
