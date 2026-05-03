CylinderFit 2026
================

**Fast, robust 3-D cylinder fitting for point clouds.**

CylinderFit 2026 fits arbitrary-oriented cylinders to ``N × 3`` NumPy arrays using
PROSAC/MAGSAC-upgraded RANSAC initialization and Levenberg-Marquardt refinement
with an analytic Jacobian — no manual threshold required.

.. code-block:: python

   from cylinderfit2026 import fit_cylinder, generate_noisy_cylinder

   syn = generate_noisy_cylinder(radius=2.5, noise=0.02, outlier_fraction=0.15)
   model = fit_cylinder(syn.points)

   print(f"radius: {model.radius:.4f}  (true: {syn.radius:.4f})")
   print(f"RMSE:   {model.rmse:.4f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api/index
   algorithm
   benchmarks_guide

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
