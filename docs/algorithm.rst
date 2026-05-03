Algorithm Notes
===============

cylfit combines three components: PROSAC/MAGSAC-upgraded RANSAC
initialization, Levenberg-Marquardt nonlinear refinement with a closed-form
analytic Jacobian, and Huber-style iteratively reweighted least squares.

RANSAC Initialization (PROSAC + MAGSAC)
----------------------------------------

Classic RANSAC draws random subsets of points, fits a model to each, and
counts inliers at a hard threshold. cylfit improves on this in
two ways:

**PROSAC sampling.** Points are ranked by their residual under the initial PCA
model (low residual = likely inlier). The first half of trials sample from an
expanding top-:math:`k` pool, so good seeds are found early. The second half
reverts to uniform sampling so outlier-dense regions can still contribute.

**MAGSAC soft scoring.** Instead of a binary inlier count at threshold
:math:`\delta`, each candidate is scored by a Gaussian-weighted sum:

.. math::

   S = \sum_i \exp\!\left(-\frac{r_i^2}{2\sigma^2}\right), \quad
   \sigma = \delta / 3

This marginalizes the threshold and makes scoring robust to its exact value —
the core idea of MAGSAC (Barath & Matas, 2019).

Levenberg–Marquardt Refinement
-------------------------------

Given a good seed :math:`(\mathbf{p}_0, \mathbf{d}, R)` from RANSAC, the
6-parameter axis + position is refined by Levenberg–Marquardt minimizing
the sum of squared radial residuals:

.. math::

   r_i = \|\mathbf{q}_i\| - R, \quad
   \mathbf{q}_i = (\mathbf{p}_i - \mathbf{p}_\text{eff})
                - \bigl[(\mathbf{p}_i - \mathbf{p}_\text{eff})\cdot\mathbf{d}\bigr]\mathbf{d}

where :math:`\mathbf{p}_\text{eff}` is the axis anchor projected to the
centroid plane so the problem is well-conditioned.

Analytic Jacobian
^^^^^^^^^^^^^^^^^

The partial derivatives of :math:`r_i` with respect to the 6 parameters are:

.. math::

   \frac{\partial r_i}{\partial \mathbf{p}_0} &= -\frac{\mathbf{q}_i}{\|\mathbf{q}_i\|} \\[6pt]
   \frac{\partial r_i}{\partial \hat{\mathbf{d}}} &=
       -\bigl[(\mathbf{p}_i - \mathbf{p}_0)\cdot\mathbf{d}\bigr]
       \frac{\mathbf{q}_i}{\|\mathbf{q}_i\|}

These are derived by differentiating through the centroid-projection step.
The key identity is that :math:`\mathbf{q}_i \perp \mathbf{d}`, which
collapses several cross-terms to zero, yielding the compact form above.

For fitted radius the Jacobian is corrected to marginalize the radius degree
of freedom by subtracting its (weighted) mean column.

Huber-Style Robust Refinement
------------------------------

After each LM step the residuals are re-weighted by Huber weights:

.. math::

   w_i = \min\!\left(1,\; \frac{1.345\,\sigma}{|r_i|}\right)

where :math:`\sigma` is estimated from the residual MAD. This down-weights
outliers without hard exclusion, allowing the optimizer to remain smooth.
