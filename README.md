# CIFAR-10 Uncertainty Benchmark

Comparing ways to make an image classifier's confidence more trustworthy:
deep ensembles, MC dropout, temperature scaling, and (in progress)
test-time augmentation. The question: which cheap method gives the most
honest confidence, and does it hold up when the data changes?

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
- [x] MC dropout
- [x] Temperature scaling
- [ ] Test-time augmentation
- [ ] Distribution shift (corrupted images, SVHN)
- [ ] Reliability diagrams

## Results (clean CIFAR-10, same 5,000 test images, mean of 3 seeds)
Temperature scaling was fitted on the other 5,000 test images.

| Method | Accuracy | Avg confidence | ECE | NLL | Cost |
|---|---|---|---|---|---|
| Single CNN | 90.6% | 93.1% | 0.026 | 0.290 | 1x |
| + temperature scaling | 90.6% | 90.4% | 0.008 | 0.277 | ~1x |
| + MC dropout (T=20) | 90.7% | 92.3% | 0.019 | 0.286 | ~1x |
| Deep ensemble (3 seeds) | 91.6% | 91.8% | 0.011 | 0.247 | 3x |
| Ensemble + temperature scaling | 91.6% | 91.0% | 0.010 | 0.246 | 3x |

Findings:
- Temperature scaling alone reduced ECE from about 0.026 to 0.008 across all three seeds (fitted temperatures 1.30 to 1.34), at essentially no cost.
- The ensemble improved accuracy by about 1 point and gave the best NLL, at 3x compute. Its ECE is comparable to temperature scaling's.
- MC dropout gave a smaller calibration gain; dropout is applied only before the final layer, which likely limits its effect.
- An ensemble gains almost nothing from extra temperature scaling (fitted T = 1.08), since it is already close to calibrated.

## Notebooks
- `01_baseline_cnn.ipynb`: data pipeline, model, training, baseline calibration
- `02_ensemble.ipynb`: 3-seed ensemble, MC dropout, temperature scaling comparison

## Run it
1. Open a notebook in Google Colab.
2. Set Runtime to a GPU (T4).
3. Run all cells top to bottom. The first run downloads CIFAR-10 (~170 MB).

## Limitations
- Only three seeds, so no formal confidence intervals.
- Temperature scaling was fitted on half of the test set rather than a separate validation set.
- ECE differences of a few thousandths (for example 0.008 vs 0.011) are within noise at 5,000 images.
- Small CNN, not a state-of-the-art architecture.
- Evaluated on clean test data only; distribution shift not yet tested.
