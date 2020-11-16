PyPair
======

Tables
------

These are the basic contingency tables used to analyze categorical data.

- CategoricalTable
- BinaryTable
- ConfusionMatrix

.. automodule:: pypair.table
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Associations
------------

Some of the functions here are just wrappers around the contingency tables and may be looked at as convenience methods to simply pass in data for two variables. If you need more than the specific association, you are encouraged to build the appropriate contingency table and then call upon the measures you need.

.. automodule:: pypair.association
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Decorators
----------

These are decorators.

.. automodule:: pypair.decorator
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
