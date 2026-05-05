# Synthetic-to-Real Transfer Learning on MNIST - Phase 2

##CSCI4701

## Project goal

**Research question:** Can a convolutional neural network trained exclusively on synthetically generated digit-images accurately classify real handwritten digits from the whole MNIST dataset? (50K images)

This project investigates whether deep learning models, especially CNNs, can learn useful features from entirely artifical training data and successfully generalize these learned representations to MNIST data images. Understanding this transfer capability of deep neural networks has real-life practical implications for fields where real data may be scarce or expensive, and generating synthetic data might be easier. In this case, whether such artifical datasets (that look effortless and slam-dunk) can transfer their features to real-life examples.

## Why this is not just a standard MNIST project

A standard MNIST classifier is trained on real MNIST and tests on real MNIST. Here, however, the research question is whether a CNN can learn and transfer digit features from artificial images that are generated with code. These images are specifically made to look smeared, blurry and not as rigid as MNIST images to mimic the variety in human handwriting and to allign with research purposes and compatibility with the classifier. Real MNIST training is used only as an upper-bound comparison; the main analysis of my research is the gap between synthetic validation performance and real MNIST test performance.

## Phase 2 experiment suite
### I wrote the text but used AI to create the table since it saved time

| Experiment group | What it tests | Why it matters |
|---|---|---|
| Real MNIST baseline | CNN trained on real MNIST | Upper-bound comparison |
| Basic synthetic 10k | Phase 1's synthetic training | Reproduces the initial transfer setting |
| Basic synthetic 50k | More synthetic samples | tests whether scale can improves feature transfer |
| Improved synthetic 50k | Noise, blur, stroke/intensity variation, shift, shear used | tests whether editing visuals helps |
| Hybrid 5%, 10%, 20% real | Synthetic data mixed with small real MNIST images | Tests how much labeled real data is needed to handle the synthetic data |
| Rotation ablation | Rotation ranges 0, 10, 20, 35 degrees (Phase 1's) | Tests whether rotation helps robustness or does the opposite |
| Thickness ablation | thin, medium, and wide stroke ranges | Tests whether stroke width affects real-domain transfer |
| Convolutional filter visualization | First-layer filters for real vs. synthetic training | Qualitative feature analysis |

## Repository structure
### I used AI for this table to be made since I didn't know how to do this format

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


## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- Pillow
- scikit-learn
- tqdm

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


## How to interpret the results

I used the real MNIST as the upper bound for the fixed CNN architecture. The synthethic-only mode answers whether artificially generated geometric digit-like shapes can transfer useful features and generalize. The improved generator comparison indicate that `visual realism` reduces the domain gap. The hybrid experiments, that I independently ran, yet decided to eventually to exclude, answered a practical and most important question: if synthetic data is cheap but a small amount of real labeled data is available, how much does that real data help?

The final answer to that was that the final data makes a noticeable/moderate difference and especially helps for numbers like 3,8,9 which
greately struggled in Phase 1. Likely due to their similarity.

The most important numbers are not only real test accuracies. Also compare validation accuracy on the training domain against real MNIST test accuracy. A high validation score with a much lower real test score is evidence of domain shift: the model learned patterns that work for the generator but do not fully match real handwriting. But also validation accuracies against real MNISTs. Our baseline was never chance (10% for our case) but beating a higher floor value, which we suceeded in doing, eventually. 

## Milestone 2 resolution results:

1. **Increase synthetic dataset size** to 50,000+ samples ✅ 
2. **Improve synthetic generation** with more realistic variations ✅ 
3. **Test hybrid approach:** Mix small amount of real data with synthetic ✅ 
4. **Visualize learned features:** Compare filters learned from synthetic vs. real data ✅ ❌
5. **Ablation studies:** Test impact of different synthetic variations (rotation, thickness, etc.) ✅ 
6. **Baseline comparison:** Train model on real MNIST to quantify performance gap ✅ 
7. **Synthetic Digit Improvement** Improve the way, especially, how some digits look ✅ 

### Visualizations (Same as before with more figures)

The notebook generates:
1. Training curves - Loss and accuracy over epochs
2. Confusion matrix - Which digits are confused with each other (please look at it yourself, very interesting results)
3. Per-class accuracy - Performance breakdown by digit
4. Prediction examples - Correct and incorrect classifications

## Limitations and failures to discuss

1. The synthetic digit generator is mnually designed and due to the limitations of the project, the entire depth of variety of human handwriting could not be captured. Instead, something very close to it, in a slightly smeared format was reproduced.
2. MNIST digits are centered, grayscale, and low-resolution; success here does not automatically imply success on much more complex real-world digit images. (Yet, it doesn't need to, since it scales well for artifically generated digits)
3. The optimized synthetic generation has been very useful for the accuracy (usually 90%+) but too much rotation/noise/thickness variation may create unrealistic images. This may, in turn, hurt the transfer.
4. Hybrid training is not a purely synthetic experiment, anymore. I included it since it drastically improves the feature domain gap. This is because it includes very small random MNIST batches. 
5. Results may vary slightly by hardware and PyTorch/CUDA randomness.


## What I did differentely from Phase 1

This time, I acted smarter and wasted less time trying to fix every single bug, and instead patched a few things,
I asked questions on forums where I could not fix the error, I stitched many pieces of codes together, did some vibe
coding, got help from AI, and delegated not important tasks to AI. For example, the main criticism to my work was that
my codes looked poorly organized. So what did I do? I did the same thing, edited and stitched my codes to their final versions
and later used AI to make evrythign look neat. To standardize camelcase for functions, to use spacing properly, and expand 
my messy comments into more readable comments.

## Computational Requirements

- **Training time:** ~5-10 minutes on Google Colab GPU
- **Memory:** <2GB GPU memory
- **Storage:** <100MB for dataset and model
- **Cost:** $0 (free Google Colab tier sufficient)

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/
- Domain Transfer Learning: Various academic papers on synthetic-to-real transfer (all read by AI and narrated to me)

## License

Feel free to use this code for educational purposes.

## Acknowledgments

Course: CSCI 4701 Deep Learning (Spring 2026)  
Institution: ADA University

by Isgandar Jafarli
