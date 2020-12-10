.. meta::
   :description: A statistical API for bivariate association measures.
   :keywords: python, statistics, bivariate, association, categorical, binary, nominal, ordinal, continuous, ratio, interval, contingency table analysis, apache spark, spark, high performance computing, massively parallel processing, hpc, mpp, causality, symmetric, asymmetric, correlation, confusion matrix, concordance, ranking
   :robots: index, follow
   :abstract: A statistical API for bivariate association measures. There are over 130 association measures identified between the product of categorical and continuous variable types.
   :author: Jee Vang, Ph.D.
   :contact: g@oneoffcoder.com
   :copyright: One-Off Coder
   :content: global
   :generator: Sphinx
   :language: English
   :rating: general
   :reply-to: info@oneoffcoder.com
   :web_author: Jee Vang, Ph.D.
   :revisit-after: 1 days

.. PyPair documentation master file, created by
   sphinx-quickstart on Wed Nov 11 22:56:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPair
======

.. image:: _static/images/logo.png
   :align: center
   :alt: pypair logo.

PyPair is a statistical library to compute pairwise association between any two types of variables. You can use the library locally on your laptop or desktop, or, you may use it on a `Spark <https://spark.apache.org/>`_ cluster.

.. blockdiag::

   diagram {
      default_shape = roundedbox
      span_width = 32
      span_height = 20
      default_fontsize = 11
      edge_layout = normal
      orientation = landscape

      V [label = "Variable", color = pink]
      C [label = "Continuous", color = "#edfa78"]
      I [label = "Interval", color = "#def514"]
      R [label = "Ratio", color = "#def514"]
      A [label = "Categorical", color = "#e0e0e0"]
      B [label = "Binary", color ="#e4ede6"]
      N [label = "Nominal", color ="#e4ede6"]
      O [label = "Ordinal", color ="#e4ede6"]

      V -> A, C
      C -> I, R
      A -> B, N, O
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro
   quicklist
   quickstart
   deepdives
   zzz-bib

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
=====

.. image:: _static/images/ooc-logo.png
   :alt: One-Off Coder logo.

One-Off Coder is an educational, service and product company. Please visit us online to discover how we may help you achieve life-long success in your personal coding career or with your company's business goals and objectives.

- |Website_Link|
- |Facebook_Link|
- |Twitter_Link|
- |Instagram_Link|
- |YouTube_Link|
- |LinkedIn_Link|

.. |Website_Link| raw:: html

   <a href="https://www.oneoffcoder.com" target="_blank">Website</a>

.. |Facebook_Link| raw:: html

   <a href="https://www.facebook.com/One-Off-Coder-309817926496801/" target="_blank">Facebook</a>

.. |Twitter_Link| raw:: html

   <a href="https://twitter.com/oneoffcoder" target="_blank">Twitter</a>

.. |Instagram_Link| raw:: html

   <a href="https://www.instagram.com/oneoffcoder/" target="_blank">Instagram</a>

.. |YouTube_Link| raw:: html

   <a href="https://www.youtube.com/channel/UCCCv8Glpb2dq2mhUj5mcHCQ" target="_blank">YouTube</a>

.. |LinkedIn_Link| raw:: html

   <a href="https://www.linkedin.com/company/one-off-coder/" target="_blank">LinkedIn</a>

Copyright
=========

Documentation
-------------

.. raw:: html

    <embed>
    This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/" target="_blank">Creative Commons Attribution 4.0 International License</a> by <a href="https://www.oneoffcoder.com" target="_blank">One-Off Coder</a>.
    <br />
    <br />
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/" target="_blank">
        <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
    <br />
    <br />
    </embed>

Software
--------

::

    Copyright 2020 One-Off Coder

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Art
---

::

    Copyright 2020 Daytchia Vang

Citation
========

::

    @misc{oneoffcoder_pypair_2020,
    title={PyPair, A Statistical API for Bivariate Association Measures},
    url={https://github.com/oneoffcoder/py-pair},
    author={Jee Vang},
    year={2020},
    month={Nov}}

Author
======

Jee Vang, Ph.D.

- |Patreon_Link|
- |Github_Link|

.. |Patreon_Link| raw:: html

   <a href="https://www.patreon.com/vangj" target="_blank">Patreon</a>: support is appreciated

.. |Github_Link| raw:: html

   <a href="https://github.com/sponsors/vangj" target="_blank">GitHub</a>: sponsorship will help us change the world for the better

Help
====

- |Source_Link|
- |Gitter_Link|

.. |Source_Link| raw:: html

   <a href="https://github.com/oneoffcoder/py-pair" target="_blank">GitHub</a>: source code

.. |Gitter_Link| raw:: html

   <a href="https://gitter.im/dataflava/py-pair" target="_blank">Gitter</a>: chat