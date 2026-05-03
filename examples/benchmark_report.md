# CylinderFit Benchmark Report

## Summary

| method | score | ok cases | radius err | axis deg | height err | rmse | seconds | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cylinderfit2026 | 53.19 | 5/5 | 0.00404 | 0.026 | 0.06005 | 0.02117 | 0.5611 | 1.000 |
| pca_baseline | 8.06 | 5/5 | 0.09558 | 17.526 | 3.42828 | 0.02972 | 0.0874 | 0.818 |

## Cases

| case | method | ok | seconds | radius err | axis deg | height err | rmse | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean_full_scan | cylinderfit2026 | yes | 0.6196 | 0.00029 | 0.009 | 0.01198 | 0.01002 | 1.000 |
| clean_full_scan | pca_baseline | yes | 0.1119 | 0.00029 | 0.009 | 0.01200 | 0.01002 | 1.000 |
| noisy_with_outliers | cylinderfit2026 | yes | 0.7381 | 0.00012 | 0.039 | 0.12340 | 0.03433 | 1.000 |
| noisy_with_outliers | pca_baseline | yes | 0.1572 | 0.00011 | 0.033 | 0.12316 | 0.03433 | 1.000 |
| partial_arc | cylinderfit2026 | yes | 0.5405 | 0.01940 | 0.016 | 0.05961 | 0.02491 | 1.000 |
| partial_arc | pca_baseline | yes | 0.0500 | 0.09947 | 0.071 | 1.26425 | 0.03556 | 0.998 |
| short_wide | cylinderfit2026 | yes | 0.2883 | 0.00014 | 0.064 | 0.06992 | 0.02479 | 1.000 |
| short_wide | pca_baseline | yes | 0.0241 | 0.37781 | 87.515 | 15.70668 | 0.05686 | 0.092 |
| long_thin | cylinderfit2026 | yes | 0.6189 | 0.00023 | 0.002 | 0.03533 | 0.01181 | 1.000 |
| long_thin | pca_baseline | yes | 0.0938 | 0.00023 | 0.002 | 0.03533 | 0.01181 | 1.000 |
