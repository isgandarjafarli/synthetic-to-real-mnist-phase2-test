# Phase 2 Results Report

## Experiment Table

| experiment                | train_source   |   synthetic_samples |   real_fraction |   epochs |   val_accuracy |   real_test_accuracy |   generalization_gap |
|:--------------------------|:---------------|--------------------:|----------------:|---------:|---------------:|---------------------:|---------------------:|
| real_mnist_baseline       | real           |                   0 |             nan |        4 |          98.73 |                98.95 |                -0.22 |
| synthetic_basic_10k       | synthetic      |               10000 |             nan |        4 |         100    |                71.07 |                28.93 |
| synthetic_basic_50k       | synthetic      |               50000 |             nan |        4 |          99.98 |                80.96 |                19.02 |
| synthetic_improved_50k    | synthetic      |               50000 |             nan |        4 |          99.58 |                83.49 |                16.09 |
| ablation_rotation_0       | synthetic      |               20000 |             nan |        2 |          99.6  |                75.36 |                24.24 |
| ablation_rotation_10      | synthetic      |               20000 |             nan |        2 |          99.35 |                80.07 |                19.28 |
| ablation_rotation_20      | synthetic      |               20000 |             nan |        2 |          98.35 |                78.3  |                20.05 |
| ablation_rotation_35      | synthetic      |               20000 |             nan |        2 |          98    |                84.1  |                13.9  |
| ablation_thickness_thin   | synthetic      |               20000 |             nan |        2 |          99.5  |                78.29 |                21.21 |
| ablation_thickness_medium | synthetic      |               20000 |             nan |        2 |          99.1  |                81.91 |                17.19 |
| ablation_thickness_wide   | synthetic      |               20000 |             nan |        2 |          98.65 |                73.51 |                25.14 |

## Interpretation

The best real-MNIST test accuracy came from **real_mnist_baseline** with **98.95%**.

Scaling the basic synthetic dataset from 10k to 50k improved transfer by **+9.89 percentage points**.

Adding Phase 2 realism operations - blur/noise/stroke/intensity/shift/shear variation - helped by **+2.53 percentage points** relative to 50k basic synthetic data.

The remaining gap to the real-data upper bound is **15.46 percentage points**, which estimates the unresolved synthetic-to-real domain gap.

The ablation study ranged from **ablation_thickness_wide** (73.51%) to **ablation_rotation_35** (84.10%), showing that generation parameters affect transfer rather than merely synthetic validation accuracy.

## Limitations and Failures

1. The synthetic generator is still hand-coded, so it cannot fully reproduce natural motor patterns, digit-writing conventions, or pen-pressure effects.
2. MNIST is grayscale, centered, and low-resolution; conclusions should not be generalized to harder vision domains without new experiments.
3. If synthetic validation accuracy is high but real test accuracy is lower, the model has learned generator-specific shortcuts. This is a central failure mode, not just noise.
4. Full Phase 2 results depend on random seeds, number of epochs, and available hardware. The notebook fixes seeds, but small differences can still occur on GPU.
5. Hybrid experiments use small labeled real subsets from MNIST, so they no longer test purely synthetic learning; they answer a different but useful question about how much real data is needed to reduce the gap.
