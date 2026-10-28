# CIFAR-10 Uncertainty Benchmark

Comparing ways to make an image classifier's confidence more trustworthy:
deep ensembles, MC dropout, and test-time augmentation (TTA). The question:
which cheap method gives the most honest confidence, especially when the
data changes?

## Motivation
Classifiers often report high confidence even when they are wrong
(overconfident). In high-stakes settings, a model should know when it is
unsure. This project measures that with calibration metrics and compares
simple, practical methods and their compute cost.

## Setup
- Dataset: CIFAR-10 (50,000 train / 10,000 test images, 10 classes)
- Model: small CNN (3 conv blocks with BatchNorm, ReLU, MaxPool, then dropout and a linear layer), 1,187,274 parameters
- Framework: PyTorch, trained on a free Colab T4 GPU
- Training: SGD with momentum, OneCycle learning rate, 15 epochs, random crop and horizontal flip augmentation
- Seeds: 0, 1, 2

## Metrics
- Accuracy: fraction of test images classified correctly.
- Average confidence: mean of the model's top predicted probability.
- ECE (Expected Calibration Error): gap between confidence and accuracy, averaged over 15 confidence bins. Lower is better.
- NLL (negative log-likelihood): penalises confident wrong predictions. Lower is better.

## Status
- [x] Baseline CNN
- [x] Baseline calibration (ECE, NLL, average confidence)
- [x] Deep ensemble (3 seeds)
- [ ] MC dropout
- [ ] Test-time augmentation
- [ ] Distribution shift (corrupted images, SVHN)
- [ ] Reliability diagrams

## Results so far (clean CIFAR-10 test set)
| Model | Accuracy | Avg confidence | ECE | NLL |
|---|---|---|---|---|
| Single CNN, seed 0 | 90.3% | 93.2% | 0.030 | 0.297 |
| Single CNN, seed 1 | 90.5% | 92.9% | 0.025 | 0.292 |
| Single CNN, seed 2 | 90.3% | 92.7% | 0.025 | 0.298 |
| Deep ensemble (3 seeds) | 91.5% | 91.7% | 0.010 | 0.253 |

The ensemble improves accuracy by about 1 point and brings average
confidence in line with accuracy (91.7% confidence vs 91.5% accuracy),
at 3x the training and inference compute. Single models are slightly
overconfident (about 2.5 points). The first baseline run in notebook 01
(unseeded) scored 90.3% accuracy, ECE 0.028, consistent with the seeded runs.

## Notebooks
- `01_baseline_cnn.ipynb`: data pipeline, model, training, baseline calibration
- `02_ensemble.ipynb`: 3-seed deep ensemble

## Run it
1. Open a notebook in Google Colab.
2. Set Runtime to a GPU (T4).
3. Run all cells top to bottom. The first run downloads CIFAR-10 (~170 MB).

## Limitations (so far)
- Only three seeds, so no statistical confidence intervals yet.
- ECE depends on the number of bins, and differences of a few thousandths are within noise.
- Small CNN, not a state-of-the-art architecture.
- Evaluated on clean test data only; distribution shift not yet tested.
