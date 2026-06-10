Quickstart
==========

SAFE-AI Metrics provides three main metric families:

* RGA — Rank Graduation Accuracy
* RGR — Rank Graduation Robustness
* RGE — Rank Graduation Explainability

Basic imports:

.. code-block:: python

   from safeai.rga import rga_curve
   from safeai.rgr import rgr_curve
   from safeai.rge import rge_curve

RGA evaluates ranking accuracy, RGR evaluates robustness under perturbations,
and RGE evaluates prediction preservation under feature removal or occlusion.