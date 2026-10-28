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

## Metrics
- Accuracy: fraction of test images classified correctly.
- Average confidence: mean of the model's top predicted probability.
- ECE (Expected Calibration Error): gap between confidence and accuracy, averaged over 15 confidence bins. Lower is better.
- NLL (negative log-likelihood): penalises confident wrong predictions. Lower is better.

## Status
- [x] Baseline CNN
- [x] Baseline calibration (ECE, NLL, average confidence)
- [ ] Deep ensemble (3 seeds)
- [ ] MC dropout
- [ ] Test-time augmentation
- [ ] Distribution shift (CIFAR-10-C, SVHN)
- [ ] Reliability diagrams

## Results so far (clean CIFAR-10 test set)
| Model | Accuracy | Avg confidence | ECE | NLL |
|---|---|---|---|---|
| Baseline CNN (single run) | 90.3% | 93.0% | 0.028 | 0.294 |

The baseline is slightly overconfident: it reports 93.0% confidence on
average but is right 90.3% of the time. Results are from one training run
and one seed, and may vary by a few tenths of a percent between runs.

## Run it
1. Open `01_baseline_cnn.ipynb` in Google Colab.
2. Set Runtime to a GPU (T4).
3. Run all cells top to bottom. The first run downloads CIFAR-10 (~170 MB).

## Limitations (so far)
- Single seed, so no confidence intervals yet.
- Small CNN, not a state-of-the-art architecture.
- Distribution shift not yet evaluated.
