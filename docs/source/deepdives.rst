Selected Deep Dives
===================

Let's go into some association measures in more details.

Binary association
------------------

.. list-table:: Contingency table for two binary variables

   * -
     - 1
     - 0
     - Total
   * - 1
     - a
     - b
     - a + b
   * - 0
     - c
     - d
     - c + d
   * - Total
     - a + c
     - b + d
     - p = a + b + c + d

.. list-table:: Contingency table for two binary variables :cite:`2010:choi,2020:psu-binary`
   :header-rows: 1

   * - Name
     - Coefficient
     - Emphasis
   * - Sokal-Michener :cite:`2010:choi`
     - :math:`\frac{a + d}{p}`
     - Equal weights for 11 and 00 matches
   * - Sørensen–Dice coefficient :cite:`2020:wiki-dice`
     - :math:`\frac{2(a + d)}{2(a + d) + b + c}`
     - Double weights for 11 and 00 matches
   * - Roger-Tanimoto :cite:`2010:choi`
     - :math:`\frac{a + d}{a + d + 2(b + c)}`
     - Double weights for mismatches
   * - Russell-Rao coefficient :cite:`2010:choi`
     - :math:`\frac{a}{p}`
     - Proportion of 11 matches
   * - Jaccard similarity coefficent :cite:`2020:wiki-jaccard`
     - :math:`\frac{a}{a + b + c}`
     - Ignores 00 matches
   * - Jaccard distance :cite:`2020:wiki-jaccard`
     - :math:`\frac{b + c}{a + b + c}`
     -
   * - Czekanowski :cite:`2010:choi`
     - :math:`\frac{2a}{2a + b + c}`
     - Ignores 00 matches and double weights for 11 matches
   * - Soka-Sneath-I :cite:`2010:choi`
     - :math:`\frac{a}{a + 2(b + c)}`
     - Ignores 00 matches and double weights for mismatches
   * - Tanimoto similarity :cite:`2020:wiki-jaccard`
     - :math:`\frac{a}{b + c}`
     - Ratio of 11 matches to mismatches ignoring 00
   * - Ochiai :cite:`2020:stack-sim`
     - :math:`\sqrt{\frac{a}{a + b}\frac{a}{a + c}}`
     -
   * - Kulczynski-2 :cite:`2020:stack-sim`
     - :math:`\frac{1}{2}\left(\frac{a}{a + b} + \frac{a}{a + c}\right)`
     -

Concordant, discordant, tie
---------------------------

Let's try to understand how to determine if a pair of observations are concordant, discordant or tied. We have made up an example dataset below having two variables :math:`X` and :math:`Y`. Note that there are 6 observations, and as such, each observation is associated with an index from 1 to 6. An observation has a pair of values, one for :math:`X` and one for :math:`Y`.

.. warning::
    Do **not** get the `pair of values of an observation` confused with a `pair of observations`.

.. list-table:: Raw Data for :math:`X` and :math:`Y`
   :header-rows: 1

   * - Index
     - :math:`X`
     - :math:`Y`
   * - 1
     - 1
     - 3
   * - 2
     - 1
     - 3
   * - 3
     - 2
     - 4
   * - 4
     - 0
     - 2
   * - 5
     - 0
     - 4
   * - 6
     - 2
     - 2

Because there are 6 observations, there are :math:`{{6}\choose{2}} = 15` possible pairs of observations. If we denote an observation by its corresponding index as :math:`O_i`, then the observations are then as follows.

- :math:`O_1 = (1, 3)`
- :math:`O_2 = (1, 3)`
- :math:`O_3 = (2, 4)`
- :math:`O_4 = (0, 2)`
- :math:`O_5 = (0, 4)`
- :math:`O_6 = (2, 2)`

The 15 possible `combinations` of observation pairings are as follows.

- :math:`O_1, O_2`
- :math:`O_1, O_3`
- :math:`O_1, O_4`
- :math:`O_1, O_5`
- :math:`O_1, O_6`
- :math:`O_2, O_3`
- :math:`O_2, O_4`
- :math:`O_2, O_5`
- :math:`O_2, O_6`
- :math:`O_3, O_4`
- :math:`O_3, O_5`
- :math:`O_3, O_6`
- :math:`O_4, O_5`
- :math:`O_4, O_6`
- :math:`O_5, O_6`

For each one of these observation pairs, we can determine if such a pair is concordant, discordant or tied. There's a couple ways to determine concordant, discordant or tie status. The easiest way to determine so is mathematically. Another way is to use rules. Both are equivalent. Because we will use abstract notation to describe these math and rules used to determine concordant, discordant or tie for each pair, and because we are striving for clarity, let's expand these observation pairs into their component pairs of values and also their corresponding :math:`X` and :math:`Y` indexed notation.

- :math:`O_1, O_2 = (1, 3), (1, 3) = (X_1, Y_1), (X_2, Y_2)`
- :math:`O_1, O_3 = (1, 3), (2, 4) = (X_1, Y_1), (X_3, Y_3)`
- :math:`O_1, O_4 = (1, 3), (0, 2) = (X_1, Y_1), (X_4, Y_4)`
- :math:`O_1, O_5 = (1, 3), (0, 4) = (X_1, Y_1), (X_5, Y_5)`
- :math:`O_1, O_6 = (1, 3), (2, 2) = (X_1, Y_1), (X_6, Y_6)`
- :math:`O_2, O_3 = (1, 3), (2, 4) = (X_2, Y_2), (X_3, Y_3)`
- :math:`O_2, O_4 = (1, 3), (0, 2) = (X_2, Y_2), (X_4, Y_4)`
- :math:`O_2, O_5 = (1, 3), (0, 4) = (X_2, Y_2), (X_5, Y_5)`
- :math:`O_2, O_6 = (1, 3), (2, 2) = (X_2, Y_2), (X_6, Y_6)`
- :math:`O_3, O_4 = (2, 4), (0, 2) = (X_3, Y_3), (X_4, Y_4)`
- :math:`O_3, O_5 = (2, 4), (0, 4) = (X_3, Y_3), (X_5, Y_5)`
- :math:`O_3, O_6 = (2, 4), (2, 2) = (X_3, Y_3), (X_6, Y_6)`
- :math:`O_4, O_5 = (0, 2), (0, 4) = (X_4, Y_4), (X_5, Y_5)`
- :math:`O_4, O_6 = (0, 2), (2, 2) = (X_4, Y_4), (X_6, Y_6)`
- :math:`O_5, O_6 = (0, 4), (2, 2) = (X_5, Y_5), (X_6, Y_6)`

Now we can finally attempt to describe how to determine if any pair of observations is concordant, discordant or tied. If we want to use math to determine so, then, for any two pairs of observations :math:`(X_i, Y_i)` and :math:`(X_j, Y_j)`, the following determines the status.

- concordant if :math:`(X_j - X_i)(Y_j - Y_i) > 0`
- discordant if :math:`(X_j - X_i)(Y_j - Y_i) < 0`
- tied if :math:`(X_j - X_i)(Y_j - Y_i) = 0`

If we like rules, then the following determines the status.

- concordant if :math:`X_i < X_j` and :math:`Y_i < Y_j` **or** :math:`X_i > X_j` and :math:`Y_i > Y_j`
- discordant if :math:`X_i < X_j` and :math:`Y_i > Y_j` **or** :math:`X_i > X_j` and :math:`Y_i < Y_j`
- tied if :math:`X_i = X_j` **or** :math:`Y_i = Y_j`

All pairs of observations will evaluate categorically to one of these statuses. Continuing with our dummy data above, the concordancy status of the 15 pairs of observations are as follows (where concordant is C, discordant is D and tied is T).

.. list-table:: Concordancy Status
   :header-rows: 1

   * - :math:`(X_i, Y_i)`
     - :math:`(X_j, Y_j)`
     - status
   * - :math:`(1, 3)`
     - :math:`(1, 3)`
     - T
   * - :math:`(1, 3)`
     - :math:`(2, 4)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 2)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 4)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 2)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 4)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 2)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 4)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 2)`
     - D
   * - :math:`(2, 4)`
     - :math:`(0, 2)`
     - C
   * - :math:`(2, 4)`
     - :math:`(0, 4)`
     - C
   * - :math:`(2, 4)`
     - :math:`(2, 2)`
     - T
   * - :math:`(0, 2)`
     - :math:`(0, 4)`
     - T
   * - :math:`(0, 2)`
     - :math:`(2, 2)`
     - T
   * - :math:`(0, 4)`
     - :math:`(2, 2)`
     - D

In this data set, the counts are :math:`C=6`, :math:`D=5` and :math:`T=4`. If we divide these counts with the total of pairs of observations, then we get the following probabilities.

- :math:`\pi_C = \frac{C}{{n}\choose{2}} = \frac{6}{15}`
- :math:`\pi_D = \frac{D}{{n}\choose{2}} = \frac{5}{15}`
- :math:`\pi_T = \frac{T}{{n}\choose{2}} = \frac{4}{15}`

Sometimes, it is desirable to distinguish between the types of ties. There are three possible types of ties.

- :math:`T^X` are ties on only :math:`X`
- :math:`T^Y` are ties on only :math:`Y`
- :math:`T^{XY}` are ties on both :math:`X` and :math:`Y`

Note, :math:`T = T^X + T^Y + T^{XY}`. If we want to distinguish between the tie types, then the status of each pair of observations is as follows.

.. list-table:: Concordancy Status
   :header-rows: 1

   * - :math:`(X_i, Y_i)`
     - :math:`(X_j, Y_j)`
     - status
   * - :math:`(1, 3)`
     - :math:`(1, 3)`
     - :math:`T^{XY}`
   * - :math:`(1, 3)`
     - :math:`(2, 4)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 2)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 4)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 2)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 4)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 2)`
     - C
   * - :math:`(1, 3)`
     - :math:`(0, 4)`
     - D
   * - :math:`(1, 3)`
     - :math:`(2, 2)`
     - D
   * - :math:`(2, 4)`
     - :math:`(0, 2)`
     - C
   * - :math:`(2, 4)`
     - :math:`(0, 4)`
     - C
   * - :math:`(2, 4)`
     - :math:`(2, 2)`
     - :math:`T^X`
   * - :math:`(0, 2)`
     - :math:`(0, 4)`
     - :math:`T^X`
   * - :math:`(0, 2)`
     - :math:`(2, 2)`
     - :math:`T^Y`
   * - :math:`(0, 4)`
     - :math:`(2, 2)`
     - D

Distinguishing between ties, in this data set, the counts are :math:`C=6`, :math:`D=5`, :math:`T^X=2`, :math:`T^Y=1` and :math:`T^{XY}=1`. The probabilities of these statuses are as follows.

- :math:`\pi_C = \frac{C}{{n}\choose{2}} = \frac{6}{15}`
- :math:`\pi_D = \frac{D}{{n}\choose{2}} = \frac{5}{15}`
- :math:`\pi_{T^X} = \frac{T^X}{{n}\choose{2}} = \frac{2}{15}`
- :math:`\pi_{T^Y} = \frac{T^Y}{{n}\choose{2}} = \frac{1}{15}`
- :math:`\pi_{T^{XY}} = \frac{T^{XY}}{{n}\choose{2}} = \frac{1}{15}`

There are quite a few measures of associations using concordance as the basis for strength of association.

.. list-table:: Association measures using concordance
   :header-rows: 1

   * - Association Measure
     - Formula
   * - Goodman-Kruskal's :math:`\gamma`
     - :math:`\gamma = \frac{\pi_C - \pi_D}{1 - \pi_T}`
   * - Somer's :math:`d`
     - :math:`d_{Y \cdot X} = \frac{\pi_C - \pi_D}{\pi_C + \pi_D + \pi_{T^Y}}`
   * -
     - :math:`d_{X \cdot Y} = \frac{\pi_C - \pi_D}{\pi_C + \pi_D + \pi_{T^X}}`
   * - Kendall's :math:`\\tau`
     - :math:`\tau = \frac{C - D}{{n}\choose{2}}`

.. note::
    Sometimes `Somers' d` is written as `Somers' D`, `Somers' Delta` or even incorrectly as `Somer's D` :cite:`2017:glen,2020:wiki-somersd`. Somers' d has two versions, one that is symmetric and one that is asymmetric. The asymmetric Somers' d is the one most typically referred to :cite:`2017:glen`. The definition of Somers' d presented here is the asymmetric one, which explains :math:`d_{Y \cdot X}` and :math:`d_{X \cdot Y}`.

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

Furthermore, :math:`\lambda` can be used in studies of causality :cite:`1983:liebetrau`. We are not saying it is appropriate or even possible to entertain causality with just two variables alone :cite:`2020:pearl,2016:pearl,2009:pearl,1988:pearl`, but, when we have two categorical variables and want to know which is likely the cause and which the effect, the asymmetry between :math:`\lambda_{A|B}` and :math:`\lambda_{B|A}` may prove informational. Causal analysis based on two variables alone has been studied :cite:`2008:nips`.
