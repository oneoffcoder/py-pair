Introduction
============

PyPair is a statistical library to compute pairwise association between any two variables. A reasonable taxonomy of variable types in statistics is as follows :cite:`2020:uom,2020:idre,2020:laerd,2020:graphpad`.

- Categorical: A variable whose values have no intrinsic ordering. An example is a variable indicating the continents: North America, South America, Asia, Arctic, Antarctica, Africa and Europe. There is no ordering to these continents; we cannot say North America comes before Africa. Categorical variables are also referred to as `qualitative` variables.
    - Binary: A categorical variable that has only 2 values. An example is a variable indicating whether or not someone likes to eat pizza; the values could be ``yes`` or ``no``. It is common to encode the binary values to ``0`` and ``1`` for storage and numerical convenience, but do not be fooled, there is still no numerical ordering. These variables are also referred to in the wild as `dichotomous` variables.
    - Nominal: A categorical variable that has 3 or more values. When most people think of categorical variables, they think of nominal variables.
    - Ordinal: A categorical variable whose values have a logical order but the difference between any two values do not give meaningful numerical magnitude. An example of an ordinal variable is one that indicates the performance on a test: good, better, best. We know that good is the base, better is the comparative and best is the superlative, however, we cannot say that the difference between best and good is two numbers up. For all we know, best can be orders of magnitude away from good.
- Continuous: A variable whose values are (basically) numbers, and thus, have meaningful ordering. A continuous variable may have an infinite number of values. Continuous variables are also referred to as `quantitative` variables.
    - Interval: A continuous variable that is one whose values exists on a continuum of numerical values. Temperature measured in Celcius or Fahrenheit is an example of an interval variable.
    - Ratio: An interval variable with a true zero. Temperature measured in Kelvin is an example of a ratio variable.

.. note::
    If we have a variable capturing eye colors, the possible values may be blue, green or brown. On first sight, this variable may be considered a nominal variable. Instead of capturing the eye color categorically, what if we measure the wavelengths of eye colors? Below are estimations of each of the wavelengths (nanometers) corresponding to these colors.

    - blue: 450
    - green: 550
    - brown: 600

    Which variable type does the eye color variable become?

.. note::
    There is also much use of the term ``discrete variable``, and sometimes it refers to categorical or continuous variables. In general, a discrete variable has a finite set of values, and in this sense, a discrete variable could be a categorical variable. We have seen many cases of a continuous variable (infinite values) undergoing `discretization` (finite values). The resulting variable from discretization is often treated as a categorical variable by applying statistical operations appropriate for that type of variable. Yet, in some cases, a continuous variable can also be a discrete variable. If we have a variable to capture age (whole numbers only), we might observe a range :math:`[0, 120]`. There are  121 values (zero is included), but still, we can treat this age variable like a ratio variable.

Assuming we have data and we know the variable types in this data using the taxonomy above, we might want to make a progression of analyses from univariate, bivariate and to multivariate analyses. Along the way, for bivariate analysis, we are often curious about the association between any two pairs of variables. We want to know both the magnitude (the strength, is it small or big?) and direction (the sign, is it positive or negative?) of the association. When the variables are all of the same type, association measures may be abound to conduct pairwise association; if all the variables are continuous, we might just want to apply canonical Pearson correlation.

The tough situation is when we have a mixed variable type of dataset; and this tough situation is quite often the normal situation. How do we find the association between a continuous and categorical variable? We can create a table as below to map the available association measures for any two types of variables :cite:`2020:calkins`. (In the table below, we collapse all continuous variable types into one).

.. raw:: html

    <table class="rc-headers">
        <tr>
            <td class="rc-headers"></td>
            <td class="rc-headers heading">Binomial</td>
            <td class="rc-headers heading">Nominal</td>
            <td class="rc-headers heading">Ordinal</td>
            <td class="rc-headers heading">Continuous</td>
        </tr>
        <tr>
            <td class="rc-headers heading">Binomial</td>
            <td class="rc-headers">
                Jaccard, Dice, Yule, Russell-Rao, Sokal-Michener, Rogers-Tanimoto, Kulzinsky, Phi
            </td>
            <td class="rc-headers">-</td>
            <td class="rc-headers">-</td>
            <td class="rc-headers">-</td>
        </tr>
        <tr>
            <td class="rc-headers heading">Nominal</td>
            <td class="rc-headers">Phi</td>
            <td class="rc-headers">Phi, L, C, Lambda</td>
            <td class="rc-headers">-</td>
            <td class="rc-headers">-</td>
        </tr>
        <tr>
            <td class="rc-headers heading">Ordinal</td>
            <td class="rc-headers">Phi</td>
            <td class="rc-headers">Rank biserial</td>
            <td class="rc-headers">Spearman rho</td>
            <td class="rc-headers">-</td>
        </tr>
        <tr class="rc-headers">
            <td class="rc-headers heading">Continuous</td>
            <td class="rc-headers">Point-biserial</td>
            <td class="rc-headers">Point-biserial</td>
            <td class="rc-headers">Biserial</td>
            <td class="rc-headers">Pearson, Kendall, Spearman, Cosine</td>
        </tr>
    </table>

The ultimate goal of this project is to identify as many measures of associations for these unique pairs of variable types and to implement these association measures in a unified application programming interface (API).

.. note::
    We use the term `association` over `correlation` since the latter typically connotes canonical Pearson correlation or association between two continuous variables. The term `association` is more general and can cover specific types of association, such as `agreement` measures, along side with those dealing with continuous variables :cite:`1983:liebetrau`.


.. bibliography:: refs.bib