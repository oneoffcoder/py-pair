Selected Deep Dives
===================

Let's go into some association measures in more details.

Goodman-Kruskal's :math:`\lambda`
---------------------------------

Goodman-Kruskal's lambda :math:`\lambda_{A|B}` measures the `proportional reduction in error` ``PRE`` for two categorical variables, :math:`A` and :math:`B`, when we want to understand how knowing :math:`B` reduces the probability of an error in predicting :math:`A`. :math:`\lambda_{A|B}` is estimated as follows.

:math:`\lambda_{A|B} = \frac{P_E - P_{E|B}}{P_E}`

Where,

- :math:`P_E = 1 - \frac{\max_c N_{+c}}{N_{++}}`
- :math:`P_{E|B} = 1 - \frac{\sum_r \max_c N_{rc}}{N_{++}}`

In meaningful language.

- :math:`P_E` is the probability of an error in predicting :math:`A`
- :math:`P_{E|B}` is the probability of an error in predicting :math:`A` given knowledge of :math:`B`

The terms :math:`N_{+c}`, :math:`N_{rc}` and :math:`N_{++}` comes from the contingency table we build from :math:`A` and :math:`B` (:math:`A` is in the columns and :math:`B` is in the rows) and denote the column marginal for the `c-th` column, total count for the `r-th` and `c-th` cell and total, correspondingly. To be clear.

- :math:`N_{+c}` is the column marginal for the `c-th` column
- :math:`N_{rc}` is total count for the `r-th` and `c-th` cell
- :math:`N_{++}` is total number of observations

The contingency table induced with :math:`A` in the columns and :math:`B` in the rows will look like the following. Note that :math:`A` has `C` columns and :math:`B` has `R` rows, or, in other words, :math:`A` has `C` values and :math:`B` has `R` values.

.. list-table:: Contingency Table for :math:`A` and :math:`B`

   * -
     - :math:`A_1`
     - :math:`A_2`
     - :math:`\dotsb`
     - :math:`A_C`
   * - :math:`B_1`
     - :math:`N_{11}`
     - :math:`N_{12}`
     - :math:`\dotsb`
     - :math:`N_{1C}`
   * - :math:`B_2`
     - :math:`N_{21}`
     - :math:`N_{22}`
     - :math:`\dotsb`
     - :math:`N_{2C}`
   * - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     -
     - :math:`\vdots`
   * - :math:`B_R`
     - :math:`N_{R1}`
     - :math:`N_{R2}`
     - :math:`\dotsb`
     - :math:`N_{RC}`

The table above only shows the cell counts :math:`N_{11}, N_{12}, \ldots, N_{RC}` and **not** the row and column marginals. Below, we expand the contingency table to include

- the row marginals :math:`N_{1+}, N_{2+}, \ldots, N_{R+}`, as well as,
- the column marginals :math:`N_{+1}, N_{+2}, \ldots, N_{+C}`.

.. list-table:: Contingency Table for :math:`A` and :math:`B`

   * -
     - :math:`A_1`
     - :math:`A_2`
     - :math:`\dotsb`
     - :math:`A_C`
     -
   * - :math:`B_1`
     - :math:`N_{11}`
     - :math:`N_{12}`
     - :math:`\dotsb`
     - :math:`N_{1C}`
     - :math:`N_{1+}`
   * - :math:`B_2`
     - :math:`N_{21}`
     - :math:`N_{22}`
     - :math:`\dotsb`
     - :math:`N_{2C}`
     - :math:`N_{2+}`
   * - :math:`\vdots`
     - :math:`\vdots`
     - :math:`\vdots`
     -
     - :math:`\vdots`
     - :math:`\vdots`
   * - :math:`B_R`
     - :math:`N_{R1}`
     - :math:`N_{R2}`
     - :math:`\dotsb`
     - :math:`N_{RC}`
     - :math:`N_{R+}`
   * -
     - :math:`N_{+1}`
     - :math:`N_{+2}`
     - :math:`\dotsb`
     - :math:`N_{+C}`
     - :math:`N_{++}`

Note that the row marginal for a row is the sum of the values across the columns, and the column marginal for a colum is the sum of the values down the rows.

- :math:`N_{R+} = \sum_C N_{RC}`
- :math:`N_{+C} = \sum_R N_{RC}`

Also, :math:`N_{++}` is just the sum over all the cells (excluding the row and column marginals). :math:`N_{++}` is really just the sample size.

- :math:`N_{++} = \sum_R \sum_C N_{RC}`

Let's go back to computing :math:`P_E` and :math:`P_{E|B}`.

:math:`P_E` is given as follows.

- :math:`P_E = 1 - \frac{\max_c N_{+c}}{N_{++}}`

:math:`\max_c N_{+c}` is returning the maximum of the column marginals, and :math:`\frac{\max_c N_{+c}}{N_{++}}` is just a probability. What probability is this one? It is the largest probability associated with a value of :math:`A` (specifically, the value of :math:`A` with the largest counts). If we were to predict which value of :math:`A` would show up, we would choose the value of :math:`A` with the highest probability (it is the most likely). We would be correct :math:`\frac{\max_c N_{+c}}{N_{++}}` percent of the time, and we would be wrong :math:`1 - \frac{\max_c N_{+c}}{N_{++}}` percent of the time. Thus, :math:`P_E` is the error in predicting :math:`A` (knowing nothing else other than the distribution, or `probability mass function` ``PMF`` of :math:`A`).

:math:`P_{E|B}` is given as follows.

- :math:`P_{E|B} = 1 - \frac{\sum_r \max_c N_{rc}}{N_{++}}`

What is :math:`\max_c N_{rc}` giving us? It is giving us the maximum cell count for the `r-th` row. :math:`\sum_r \max_c N_{rc}` adds up the all the largest values in each row, and :math:`\frac{\sum_r \max_c N_{rc}}{N_{++}}` is again a probability. What probability is this one? This probability is the one associated with predicting the value of :math:`A` when we know :math:`B`. When we know what the value of :math:`B` is, then the value of :math:`A` should be the one with the largest count (it has the highest probability, or, equivalently, the highest count). When we know the value of :math:`B` and by always choosing the value of :math:`A` with the highest count associated with that value of :math:`B`, we are correct :math:`\frac{\sum_r \max_c N_{rc}}{N_{++}}` percent of the time and incorrect :math:`1 - \frac{\sum_r \max_c N_{rc}}{N_{++}}` percent of the time. Thus, :math:`P_{E|B}` is the error in predicting :math:`A` when we know the value of :math:`B` and the PMF of :math:`A` given :math:`B`.

The expression :math:`P_E - P_{E|B}` is the reduction in the probability of an error in predicting :math:`A` given knowledge of :math:`B`. This expression represents the `reduction in error` in the phrase/term ``PRE``. The proportional part in ``PRE`` comes from the expression :math:`\frac{P_E - P_{E|B}}{P_E}`, which is a proportion.

What :math:`\lambda_{A|B}` is trying to compute is the reduction of error in predicting :math:`A` when we know :math:`B`. Did we reduce any prediction error of :math:`A` by knowing :math:`B`?

- When :math:`\lambda_{A|B} = 0`, this value means that knowing :math:`B` did not reduce any prediction error in :math:`A`. The only way to get :math:`\lambda_{A|B} = 0` is when :math:`P_E = P_{E|B}`.
- When :math:`\lambda_{A|B} = 1`, this value means that knowing :math:`B` completely reduced all prediction errors in :math:`A`. The only way to get :math:`\lambda_{A|B} = 1` is when :math:`P_{E|B} = 0`.

Generally speaking, :math:`\lambda_{A|B} \neq \lambda_{B|A}`, and :math:`\lambda` is thus an asymmetric association measure. To compute :math:`\lambda_{B|A}`, simply put :math:`B` in the columns and :math:`A` in the rows and reuse the formulas above.

Furthermore, :math:`\lambda` can be used in studies of causality :cite:`1983:liebetrau`. We are not saying it is appropriate or even possible to entertain causality with just two variables alone :cite:`2020:pearl,2016:pearl,2009:pearl,1988:pearl`, but, when we have two categorical variables and want to know which is likely the cause and which the effect, the asymmetry between :math:`\lambda_{A|B}` and :math:`\lambda_{B|A}` may prove informational.
