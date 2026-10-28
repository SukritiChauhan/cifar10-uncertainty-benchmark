# CIFAR-10 Uncertainty Benchmark

Comparing ways to make an image classifier's confidence more trustworthy:
deep ensembles, MC dropout, and test-time augmentation (TTA). The question:
which cheap method gives the most honest confidence, especially when the
data changes?

## Motivation
Classifiers often report high confidence even when they are wrong
(overconfident). In high-stakes settings, a model should know when it is
unsure. This project measures that with calibration metrics and compares
simple, practical methods at comparable compute.

## Setup
- Dataset: CIFAR-10 (50,000 train / 10,000 test images, 10 classes)
- Model: small CNN (3 conv blocks with BatchNorm, ReLU, MaxPool, then dropout and a linear layer), 1,187,274 parameters
- Framework: PyTorch, trained on a free Colab T4 GPU
- Training: SGD with momentum, OneCycle learning rate, 15 epochs, random crop and horizontal flip augmentation

## Status
- [x] Baseline CNN
- [ ] Baseline calibration (ECE, NLL, avg confidence)
- [ ] Deep ensemble
- [ ] MC dropout
- [ ] Test-time augmentation
- [ ] Distribution shift (CIFAR-10-C, SVHN)
- [ ] Reliability diagrams and results table

## Results so far
| Model | Test accuracy |
|---|---|
| Baseline CNN (single run) | 90.3% |

Notes: one training run, one random seed, 15 epochs. Accuracy may vary by a
few tenths of a percent between runs. Calibration results will be added next.

## Run it
1. Open `01_baseline_cnn.ipynb` in Google Colab.
2. Set Runtime to a GPU (T4).
3. Run all cells top to bottom. The first run downloads CIFAR-10 (~170 MB).

## Limitations (so far)
- Single seed, so no confidence intervals yet.
- Small CNN, not a state-of-the-art architecture.
- Calibration and distribution shift not yet evaluated.
