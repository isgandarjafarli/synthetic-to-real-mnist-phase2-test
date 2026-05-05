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

The table below is intentionally generated from actual experiment outputs. Run the notebook or the command above to replace this section with real numbers.

<!-- RESULTS_START -->
Results not generated yet. Run `python run_phase2.py --mode full --update-readme` or run all cells in `phase2_notebook.ipynb`.
<!-- RESULTS_END -->

## How to interpret the results

Use the real MNIST baseline as the upper bound for the fixed CNN architecture. The synthetic-only runs answer whether artificial geometric data transfers at all. The 10k vs 50k comparison separates dataset scale from generator quality. The improved-generator comparison tests whether visual realism reduces the domain gap. The hybrid experiments answer a practical question: if synthetic data is cheap but a small amount of real labeled data is available, how much does that real data help?

The most important number is not only real test accuracy. Also compare validation accuracy on the training domain against real MNIST test accuracy. A high validation score with a much lower real test score is evidence of domain shift: the model learned patterns that work for the generator but do not fully match real handwriting.

## Limitations and failures to discuss

1. The synthetic digit generator is mnually designed and due to the limitations of the project, the entire depth of variety of human handwriting could not be captured. Instead, something very close to it, in a slightly smeared format was reproduced.
2. MNIST digits are centered, grayscale, and low-resolution; success here does not automatically imply success on much more complex real-world digit images. (Yet, it doesn't need to, since it scales well for artifically generated digits)
3. The optimized synthetic generation has been very useful for the accuracy (usually 90%+) but too much rotation/noise/thickness variation may create unrealistic images. This may, in turn, hurt the transfer.
4. Hybrid training is not a purely synthetic experiment, anymore. I included it since it drastically improves the feature domain gap. This is because it includes very small random MNIST batches. 
5. Results may vary slightly by hardware and PyTorch/CUDA randomness.

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
