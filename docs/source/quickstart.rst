Quickstart
==========

Installation
------------

Use PyPi to install the `package <https://pypi.org/project/pypair>`_.

.. code:: bash

    pip install pypair

Confusion Matrix
----------------

A confusion matrix is typically used to judge binary classification performance. There are two variables, :math:`A` and :math:`P`, where :math:`A` is the actual value (ground truth) and :math:`P` is the predicted value. The example below shows how to use the convenience method ``confusion()`` and the class ``ConfusionMatrix`` to get association measures derived from the confusion matrix.

.. literalinclude:: _code/confusion-demo.py
   :language: python
   :linenos:

Binary-Binary
-------------

Association measures for binary-binary variables are computed using ``binary_binary()`` or ``BinaryTable``.

.. literalinclude:: _code/binary-demo.py
   :language: python
   :linenos:

Categorical-Categorical
-----------------------

Association measures for categorical-categorical variables are computed using ``categorical_categorical()`` or ``CategoricalTable``.

.. literalinclude:: _code/binary-demo.py
   :language: python
   :linenos:

Binary-Continuous
-----------------

Association measures for binary-continuous variables are computed using ``binary_continuous()`` or ``Biserial``.

.. literalinclude:: _code/biserial-demo.py
   :language: python
   :linenos:

Ordinal-Ordinal, Concordance
----------------------------
Concordance measures are used for ordinal-ordinal or continuous-continuous variables using ``concordance()`` or ``Concordance()``.

.. literalinclude:: _code/concordance-demo.py
   :language: python
   :linenos:

Categorical-Continuous
----------------------
Categorical-continuous association measures are computed using ``categorical_continuous()`` or ``CorrelationRatio``.

.. literalinclude:: _code/corr-ratio-demo.py
   :language: python
   :linenos:

Continuous-Continuous
---------------------

Association measures for continuous-continuous variables are computed using ``continuous_continuous()`` or ``Continuous``.

.. literalinclude:: _code/continuous-demo.py
   :language: python
   :linenos: