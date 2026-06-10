Installation
============

Install from PyPI:

.. code-block:: bash

   pip install safe-ai-metrics

For development, install the package from the project root:

.. code-block:: bash

   pip install -e .

Then import the metric modules:

.. code-block:: python

   from safeai.rga import rga_score, rga_curve, compare_rga
   from safeai.rgr import rgr_score, rgr_curve, compare_rgr
   from safeai.rge import rge_score, rge_curve, compare_rge