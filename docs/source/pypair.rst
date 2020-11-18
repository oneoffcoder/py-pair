PyPair
======

Contigency Table Analysis
-------------------------

These are the basic contingency tables used to analyze categorical data.

- CategoricalTable
- BinaryTable
- ConfusionMatrix
- AgreementTable

.. automodule:: pypair.contigency
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Biserial
--------

These are the biserial association measures.

.. automodule:: pypair.biserial
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Continuous
----------

These are the continuous association measures.

.. automodule:: pypair.continuous
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

Utility
-------

These are utility functions.

.. automodule:: pypair.util
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Spark
-----

These are functions that you can use in a Spark.

.. automodule:: pypair.spark
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
