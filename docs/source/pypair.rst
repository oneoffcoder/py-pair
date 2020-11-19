PyPair
======

Contingency Table Analysis
--------------------------

These are the basic contingency tables used to analyze categorical data.

- CategoricalTable
- BinaryTable
- ConfusionMatrix
- AgreementTable

.. automodule:: pypair.contingency
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

These are functions that you can use in a Spark. You must pass in a Spark dataframe and you will get a ``pair-RDD`` as output. The pair-RDD will have the following as its keys and values.

- key: in the form of a tuple of strings ``(k1, k2)`` where k1 and k2 are names of variables (column names)
- value: a dictionary ``{'acc': 0.8, 'tpr': 0.9, 'fpr': 0.8, ...}`` where keys are association measure names and values are the corresponding association values



.. automodule:: pypair.spark
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
