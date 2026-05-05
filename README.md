# Synthetic-to-Real Transfer Learning on MNIST - Phase 2

## Project goal

**Research question:** Can a convolutional neural network trained on synthetically generated digit images classify real handwritten MNIST digits, and which changes to the synthetic data generation process reduce the synthetic-to-real domain gap?

This project studies a focused deep-learning transfer problem. The model is not changed between experiments. Instead, the training data changes: basic synthetic digits, larger synthetic datasets, improved synthetic digits with noise/blur/stroke variation, small real-data hybrids, and controlled generation ablations. The target evaluation is always real MNIST test accuracy.

## Why this is not just a standard MNIST project

A standard MNIST classifier trains on real MNIST and tests on real MNIST. Here, the central question is whether a CNN can learn transferable digit features from artificial images that are generated with code. Real MNIST training is used only as an upper-bound comparison. The main analysis is the gap between synthetic validation performance and real MNIST test performance.

## Phase 2 experiment suite

| Experiment group | What it tests | Why it matters |
|---|---|---|
| Real MNIST baseline | CNN trained on real MNIST | Upper-bound comparison for the same model architecture |
| Basic synthetic 10k | Phase 1-style synthetic training | Reproduces the initial transfer setting |
| Basic synthetic 50k | More synthetic samples | Tests whether scale alone improves transfer |
| Improved synthetic 50k | Noise, blur, stroke/intensity variation, shift, shear | Tests whether reducing the visual domain gap helps |
| Hybrid 5%, 10%, 20% real | Synthetic data mixed with small real MNIST subsets | Tests how much labeled real data is needed to anchor the synthetic data |
| Rotation ablation | Rotation ranges 0, 10, 20, 35 degrees | Tests whether rotation helps robustness or creates unrealistic examples |
| Thickness ablation | Thin, medium, and wide stroke ranges | Tests whether stroke width affects real-domain transfer |
| Conv-filter visualization | First-layer filters for real vs synthetic training | Qualitative feature analysis |

## Repository structure

```text
synthetic-to-real-mnist/
├── README.md
├── phase2_notebook.ipynb            # Main graded Milestone 2 notebook
├── milestone1_notebook.ipynb        # Original Phase 1 notebook kept for continuity
├── run_phase2.py                    # Command-line runner
├── requirements.txt
├── src/
│   └── synthetic_to_real_mnist/
│       ├── config.py                # Experiment configuration dataclasses
│       ├── datasets.py              # Real, synthetic, and hybrid DataLoaders
│       ├── experiments.py           # Phase 2 experiment orchestration
│       ├── model.py                 # CNN architecture
│       ├── reporting.py             # Auto-generated results report and README update
│       ├── synthetic_data.py        # Basic/improved synthetic digit generator
│       ├── train.py                 # Training and evaluation utilities
│       └── visualization.py         # Plots and feature visualizations
├── model.py                         # Backward-compatible wrapper
├── synthetic_data.py                # Backward-compatible wrapper
├── train.py                         # Backward-compatible wrapper
└── outputs/                         # Generated after running the notebook/script
```

## Team members and responsibilities

| Name | Specific contributions |
|---|---|
| Isgandar Jafarli | Project design, synthetic digit generation, PyTorch training pipeline, Phase 1 implementation, Phase 2 experiment suite, analysis, README, and notebook preparation |

If this project has teammates, add their names and exact contributions before submission.

## Setup

### Google Colab

1. Open `phase2_notebook.ipynb` in Google Colab.
2. Make sure the runtime is using a GPU: `Runtime -> Change runtime type -> GPU`.
3. Run all cells from top to bottom.
4. The notebook will clone this repository if needed, install dependencies, run experiments, save results under `outputs/`, and update the marked results section in this README.

### Local run

```bash
git clone https://github.com/isgandarjafarli/synthetic-to-real-mnist.git
cd synthetic-to-real-mnist
python -m pip install -r requirements.txt
python run_phase2.py --mode full --update-readme
```

For a quick technical check before the full run:

```bash
python run_phase2.py --mode smoke --no-save-models
```

The smoke mode is only for debugging. Use full mode for final reported results.

## Reproducing reported results

The main graded workflow is:

```bash
python run_phase2.py --mode full --update-readme
```

This creates:

- `outputs/phase2_results.csv`
- `outputs/phase2_report.md`
- `outputs/plots/phase2_experiment_summary.png`
- one training-curve plot, confusion matrix, and per-class plot for each experiment
- convolution-filter visualizations for the real baseline and improved synthetic model

## Main results

The table below is intentionally generated from actual experiment outputs. Run the notebook or the command above to replace this section with real numbers.

<!-- RESULTS_START -->
Results not generated yet. Run `python run_phase2.py --mode full --update-readme` or run all cells in `phase2_notebook.ipynb`.
<!-- RESULTS_END -->

## How to interpret the results

Use the real MNIST baseline as the upper bound for the fixed CNN architecture. The synthetic-only runs answer whether artificial geometric data transfers at all. The 10k vs 50k comparison separates dataset scale from generator quality. The improved-generator comparison tests whether visual realism reduces the domain gap. The hybrid experiments answer a practical question: if synthetic data is cheap but a small amount of real labeled data is available, how much does that real data help?

The most important number is not only real test accuracy. Also compare validation accuracy on the training domain against real MNIST test accuracy. A high validation score with a much lower real test score is evidence of domain shift: the model learned patterns that work for the generator but do not fully match real handwriting.

## Limitations and failures to discuss

1. The synthetic generator is manually designed, so it cannot capture all natural handwriting behavior.
2. MNIST digits are centered, grayscale, and low-resolution; success here does not automatically imply success on harder real-world images.
3. Improved synthetic generation can help, but too much rotation/noise/thickness variation may create unrealistic images and hurt transfer.
4. Hybrid training is no longer a purely synthetic experiment. It is included because it gives a practical comparison for reducing the domain gap.
5. Results may vary slightly by hardware and PyTorch/CUDA nondeterminism, although seeds are fixed.

## Submission notes

For Milestone 2, submit both:

1. the public GitHub repository link, and
2. a ZIP archive of the same final repository.

Before zipping, run the full notebook/script once so `outputs/phase2_results.csv`, `outputs/phase2_report.md`, and the README results section reflect the final experiment outputs.
