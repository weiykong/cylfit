Benchmarking
============

cylfit ships a built-in benchmark framework for comparing fitting
methods head-to-head on synthetic test cases.

Running the Built-in Benchmark
-------------------------------

.. code-block:: python

   from cylfit import (
       run_benchmark,
       summarize_benchmark,
       benchmark_markdown_report,
       default_benchmark_cases,
   )

   results = run_benchmark(cases=default_benchmark_cases())
   print(summarize_benchmark(results))
   print(benchmark_markdown_report(results))

Adding Custom Cases
--------------------

.. code-block:: python

   from cylfit.benchmarks import BenchmarkCase
   from cylfit import generate_noisy_cylinder

   syn = generate_noisy_cylinder(
       radius=1.0, height=5.0, noise=0.02, outlier_fraction=0.3
   )
   cases = [BenchmarkCase("my_hard_case", syn, threshold=0.05)]
   results = run_benchmark(cases=cases)

Adding Competitor Methods
--------------------------

.. code-block:: python

   from cylfit import run_benchmark, default_benchmark_cases, available_methods

   # available_methods() lists all detected competitors
   print(available_methods())

   # Pass a dict of {name: callable(points) -> (axis_point, axis_dir, radius)}
   def my_method(points):
       ...
       return axis_point, axis_direction, radius

   results = run_benchmark(
       cases=default_benchmark_cases(),
       methods={"my_method": my_method},
   )

Benchmark Metrics
-----------------

Each result includes:

* **radius_error** — absolute error in radius vs ground truth
* **axis_angle_deg** — angle between fitted and true axis (degrees)
* **rmse** — root mean squared inlier residual
* **inlier_recall** — fraction of true inliers correctly identified
* **runtime_s** — wall-clock time in seconds
* **score** — composite quality score (0–100)
