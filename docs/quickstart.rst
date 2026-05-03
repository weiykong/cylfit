Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install cylfit

For optional features::

   pip install cylfit[visualize]   # matplotlib plots
   pip install cylfit[dev]         # pytest, hypothesis

Basic Fitting
-------------

Pass an ``N × 3`` NumPy array and get a :class:`~cylfit.CylinderModel`
back:

.. code-block:: python

   import numpy as np
   from cylfit import fit_cylinder

   # Any N×3 array — from a sensor, file, or generated data
   points = np.load("scan.npy")
   model  = fit_cylinder(points)

   print(model.radius)
   print(model.axis_direction)
   print(model.rmse)

Loading Point Cloud Files
--------------------------

.. code-block:: python

   from cylfit import load_points, fit_cylinder

   pts   = load_points("pipe_scan.ply")   # PLY, PCD, LAS/LAZ, XYZ
   model = fit_cylinder(pts)

Using an Open3D PointCloud
--------------------------

.. code-block:: python

   import open3d as o3d
   from cylfit import from_open3d, fit_cylinder

   cloud = o3d.io.read_point_cloud("scan.ply")
   pts   = from_open3d(cloud)
   model = fit_cylinder(pts)

Constrained Fitting
-------------------

Fix the axis direction (e.g., vertical pipes):

.. code-block:: python

   from cylfit import fit_cylinder_constrained_axis
   import numpy as np

   model = fit_cylinder_constrained_axis(
       points,
       axis=[0, 0, 1],          # prefer Z-up
       max_axis_angle_deg=10.0, # allow ±10° deviation
   )

Multi-Cylinder Detection
------------------------

.. code-block:: python

   from cylfit import detect_cylinders

   detections = detect_cylinders(points, max_cylinders=5, min_inliers=300)
   for det in detections:
       print(det.model.radius, det.model.height)

Parallel RANSAC
---------------

Speed up large-cloud fitting with all CPU cores:

.. code-block:: python

   model = fit_cylinder(points, ransac_trials=256, n_jobs=-1)

Exporting Results
-----------------

.. code-block:: python

   import json

   print(model.to_json(indent=2))         # JSON string
   vertices, faces = model.to_mesh()       # triangle mesh
   model.plot(points)                      # matplotlib 3-D plot
