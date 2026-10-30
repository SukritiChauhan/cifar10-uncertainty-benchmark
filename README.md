# CIFAR-10 Uncertainty Benchmark

Comparing ways to make an image classifier's confidence more trustworthy:
deep ensembles, MC dropout, temperature scaling, test-time augmentation (TTA),
and noise/blur augmentation. The question: which cheap method gives the most
honest confidence, and does it hold up when the data changes?

## Motivation
Classifiers often report high confidence even when they are wrong
(overconfident). In high-stakes settings, a model should know when it is
unsure. This project measures that with calibration metrics and compares
simple, practical methods and their compute cost.

## Metrics
- Accuracy: fraction of test images classified correctly.
- Average confidence: mean of the model's top predicted probability.
- ECE (Expected Calibration Error): gap between confidence and accuracy, averaged over 15 confidence bins. Lower is better.
- NLL (negative log-likelihood): penalises confident wrong predictions. Lower is better.
- AUROC: how well confidence separates in-distribution from out-of-distribution images (0.5 = chance, 1.0 = perfect).

## Two protocols
- **Part A (final):** 45,000 training / 5,000 held-out validation / 10,000 test images, 5 seeds. Temperature is fitted on validation only; the test set is used for final numbers. Notebook 04.
- **Part B (earlier exploration):** all 50,000 training images, 3 seeds, temperature fitted on half of the test set and evaluated on the other half. Notebooks 01–03. Kept because it contains experiments not repeated in Part A (10-view TTA, the failed 4-pixel TTA variant, reliability diagrams).

The two protocols agree where they overlap (for example temperature scaling cuts ECE from about 0.03 to about 0.007–0.008 in both).

## Status
- [x] Baseline CNN and calibration (ECE, NLL, average confidence)
- [x] Held-out validation split and 5 seeds (Part A)
- [x] Deep ensemble, MC dropout, temperature scaling, test-time augmentation
- [x] Distribution shift: Gaussian noise, blur (both parts); contrast, brightness (Part A)
- [x] Noise/blur augmentation tested on seen and unseen corruptions (Part A)
- [x] Reliability diagrams (Part B)
  ![Reliability diagrams](reliability.png)
- [x] Out-of-distribution data (SVHN, Part A)
- [ ] Larger model (ResNet-18) and CIFAR-100
- [ ] Packaged library with tests and CI

---

# Part A: final protocol (notebook 04)

## Setup
- Dataset: CIFAR-10, 45,000 train / 5,000 validation (fixed split) / 10,000 test, 10 classes
- Model: small CNN (3 blocks of two conv + BatchNorm + ReLU, then max-pool; dropout 0.3; linear layer), 1,187,274 parameters
- Framework: PyTorch, trained on a free Colab T4 GPU
- Training: 15 epochs, SGD with Nesterov momentum, OneCycle learning rate (max 0.05), weight decay 5e-4, random crop and horizontal flip
- Seeds: 0–4
- Temperature scaling: one temperature per model, grid search on validation NLL (clean validation data only, even when evaluating on shifted data)

## Results on clean test data (mean ± std over 5 seeds)

| Method | Accuracy | Confidence | ECE | NLL |
|---|---|---|---|---|
| single model | 0.899 ± 0.003 | 0.927 ± 0.001 | 0.029 ± 0.003 | 0.312 ± 0.006 |
| single + temperature | 0.899 ± 0.003 | 0.898 ± 0.002 | 0.007 ± 0.001 | 0.296 ± 0.006 |
| MC dropout (20 passes) | 0.898 ± 0.003 | 0.919 ± 0.001 | 0.022 ± 0.002 | 0.305 ± 0.006 |
| MC dropout + temperature | 0.898 ± 0.003 | 0.896 ± 0.002 | 0.007 ± 0.001 | 0.296 ± 0.006 |
| flip TTA | 0.903 ± 0.002 | 0.919 ± 0.002 | 0.016 ± 0.002 | 0.287 ± 0.006 |
| flip TTA + temperature | 0.903 ± 0.002 | 0.900 ± 0.003 | 0.006 ± 0.002 | 0.281 ± 0.006 |
| 5-model ensemble | 0.912 | 0.912 | 0.008 | 0.257 |
| 5-model ensemble + temperature | 0.912 | 0.907 | 0.008 | 0.257 |

(The ensemble is a single 5-model ensemble, so it has no ± value.)

## Robustness to distribution shift
Models trained normally (**std**) vs. trained with noise/blur augmentation (**aug**: about 25% of each batch noised, 25% blurred). Temperatures are fitted on clean validation data. Cells show accuracy / ECE. Rows marked *unseen* are corruptions the aug models never trained on.

| Condition | std single | aug single | std ensemble | aug ensemble |
|---|---|---|---|---|
| clean | 0.899 / 0.007 | 0.871 / 0.007 | 0.912 / 0.008 | 0.882 / 0.006 |
| noise σ=0.05 | 0.646 / 0.126 | 0.857 / 0.010 | 0.669 / 0.104 | 0.870 / 0.012 |
| noise σ=0.1 | 0.305 / 0.386 | 0.826 / 0.016 | 0.307 / 0.381 | 0.846 / 0.021 |
| noise σ=0.2 | 0.138 / 0.536 | 0.730 / 0.022 | 0.119 / 0.578 | 0.759 / 0.035 |
| blur σ=0.7 | 0.704 / 0.045 | 0.855 / 0.014 | 0.725 / 0.041 | 0.867 / 0.013 |
| blur σ=1.0 | 0.406 / 0.209 | 0.832 / 0.016 | 0.419 / 0.227 | 0.849 / 0.017 |
| blur σ=1.5 | 0.277 / 0.291 | 0.793 / 0.016 | 0.285 / 0.326 | 0.809 / 0.014 |
| contrast ×0.5 (unseen) | 0.835 / 0.013 | 0.824 / 0.011 | 0.851 / 0.012 | 0.838 / 0.011 |
| contrast ×0.3 (unseen) | 0.599 / 0.037 | 0.700 / 0.016 | 0.626 / 0.034 | 0.715 / 0.021 |
| contrast ×0.15 (unseen) | 0.216 / 0.169 | 0.391 / 0.072 | 0.232 / 0.172 | 0.412 / 0.077 |
| brightness +0.3 (unseen) | 0.841 / 0.013 | 0.815 / 0.011 | 0.860 / 0.009 | 0.832 / 0.008 |

## Out-of-distribution detection (CIFAR-10 test vs. SVHN)
Score = maximum softmax probability. First 10,000 SVHN test images. Confidences are uncalibrated.

| Method | AUROC | Avg. confidence (CIFAR) | Avg. confidence (SVHN) |
|---|---|---|---|
| std, single | 0.926 ± 0.010 | 0.927 | 0.600 |
| std, 5-model ensemble | 0.925 | 0.912 | 0.567 |
| aug, single | 0.899 ± 0.003 | 0.898 | 0.570 |
| aug, 5-model ensemble | 0.907 | 0.885 | 0.532 |

## Findings (Part A)
1. **Models are overconfident, and temperature scaling is the cheapest fix.** A single model claims 92.7% average confidence at 89.9% accuracy. One fitted temperature (about 1.3) cuts ECE from 0.029 to 0.007 with no accuracy change, and it holds with a clean validation split.
2. **MC dropout adds nothing once temperature scaling is used** (ECE 0.007 for both). Flip TTA gives a small accuracy and NLL gain (+0.4 points).
3. **Ensembles give the best accuracy (+1.3 points) and NLL**, and are already well calibrated (fitted temperature about 1.05), but cost 5x compute.
4. **Under noise and blur, standard models collapse while staying overconfident** (σ=0.1 noise: 30.5% accuracy, ECE 0.386). Ensembling does not help.
5. **Noise/blur augmentation largely fixes this** (σ=0.1 noise: 82.6% accuracy, ECE 0.016) and calibration survives even though the temperature was fitted on clean data. The cost is about 2.8 points of clean accuracy.
6. **The benefit only partly transfers to unseen corruptions.** It helps for strong contrast loss (×0.3: 59.9% → 70.0%) but not for mild contrast or brightness changes, where standard models are slightly better.
7. **Augmentation slightly hurts OOD detection** (AUROC 0.926 → 0.899). Ensembling does not improve it.

---

# Part B: earlier exploration (notebooks 01–03)

Setup: all 50,000 training images, seeds 0, 1, 2, same CNN and training recipe. Temperature scaling was fitted on one half (5,000 images) of the test set and all methods are reported on the other half.

## Results on clean CIFAR-10 (mean of 3 seeds)

| Method | Accuracy | Avg confidence | ECE | NLL | Cost |
|---|---|---|---|---|---|
| Single CNN | 90.6% | 93.1% | 0.026 | 0.290 | 1x |
| + temperature scaling | 90.6% | 90.4% | 0.008 | 0.277 | ~1x |
| + MC dropout (20 passes) | 90.7% | 92.3% | 0.019 | 0.286 | ~1x training |
| + TTA, 2 views (mirror) | 91.0% | 92.2% | 0.013 | 0.266 | 2x inference |
| + TTA, 10 mild views | 91.4% | 91.5% | 0.008 | 0.252 | 10x inference |
| Deep ensemble (3 seeds) | 91.6% | 91.8% | 0.011 | 0.247 | 3x training and inference |
| Ensemble + temperature scaling | 91.6% | 91.0% | 0.010 | 0.246 | 3x |

Findings on clean data:
- Temperature scaling alone reduced ECE from about 0.026 to 0.008 across all three seeds, at essentially no cost.
- TTA with 10 mild views (shifts of up to 2 pixels plus mirrors) came within 0.2 points of the ensemble's accuracy and matched temperature scaling's ECE, with no extra training, at 10x inference cost.
- The ensemble gave the best NLL and highest accuracy, at 3x compute.
- MC dropout gave a smaller calibration gain; dropout is applied only before the final layer, which likely limits its effect.
- A first TTA variant with larger shifts (up to 4 pixels) and mismatched padding hurt: accuracy fell to about 89% and the model became underconfident (checked on seeds 0 and 1 only). TTA views should match the augmentations used in training.

## Distribution shift (mean of 3 seeds, ECE; lower is better)
Gaussian noise and blur were applied to the test images. Temperatures were fitted once on clean data and reused.

| Condition | Accuracy (single) | Single | + temp | MC dropout | TTA 2 | Ensemble | Ensemble + temp |
|---|---|---|---|---|---|---|---|
| Clean | 90.6% | 0.026 | 0.008 | 0.019 | 0.013 | 0.011 | 0.010 |
| Noise 0.05 | 62.8% | 0.198 | 0.146 | 0.186 | 0.170 | 0.140 | 0.125 |
| Noise 0.10 | 29.9% | 0.453 | 0.385 | 0.438 | 0.430 | 0.390 | 0.369 |
| Noise 0.20 | 15.8% | 0.512 | 0.428 | 0.500 | 0.490 | 0.452 | 0.427 |
| Blur 0.7 | 71.1% | 0.101 | 0.043 | 0.091 | 0.077 | 0.054 | 0.038 |
| Blur 1.0 | 41.7% | 0.270 | 0.189 | 0.259 | 0.250 | 0.213 | 0.190 |
| Blur 1.5 | 29.0% | 0.336 | 0.250 | 0.325 | 0.324 | 0.282 | 0.258 |

Findings under shift:
- Accuracy falls sharply because the model was trained on clean images only.
- Confidence falls far less than accuracy, so every method becomes badly overconfident (for example, about 75% confidence at 30% accuracy under noise 0.1).
- Temperature scaling fitted on clean data still reduces ECE (roughly 15% to 55%), but does not fix the problem.
- At heavy shift, a single model with temperature scaling is about as well calibrated as the 3-model ensemble; the ensemble does not clearly justify its 3x cost for calibration.
- Calibration fixes cannot rescue a model whose accuracy has collapsed. This motivated the noise/blur augmentation experiment in Part A.

---

## Notebooks
- `01_baseline_cnn.ipynb`: data pipeline, model, training, baseline calibration (Part B)
- `02_ensemble.ipynb`: 3-seed ensemble, MC dropout, temperature scaling (Part B)
- `03_tta_and_shift.ipynb`: test-time augmentation and distribution shift experiments (Part B)
- `04_validation_and_robustness.ipynb`: held-out validation, 5 seeds, noise/blur augmentation, unseen corruptions, SVHN OOD (Part A, final protocol)

## Run it
1. Open a notebook in Google Colab.
2. Set Runtime to a GPU (T4).
3. Run all cells top to bottom. The first run downloads CIFAR-10 (~170 MB).
4. In notebook 04, trained models are saved to Google Drive and training skips any model that already exists, so you can resume after a disconnect. Training takes roughly 4–5 minutes per model on a T4.

## Limitations
- Small CNN (1.19M parameters) on a small dataset; results may not transfer to larger models.
- Part A: 5 seeds; the ensemble is a single 5-model ensemble with no spread estimate; validation set is 5,000 images. Some ECE differences (for example 0.006 vs 0.007) are within noise.
- Part B: only three seeds, and temperature scaling was fitted on half of the test set rather than a separate validation set. The mild-shift TTA variant was designed after the first variant underperformed and evaluated on the same half, so its result involves some selection. The 10-view TTA was not repeated under the Part A protocol.
- Corruptions are synthetic (noise, blur, contrast, brightness; one noise realisation per level), not the standard CIFAR-10-C benchmark.
- The augmentation recipe was tried at one setting and not tuned. Rows where the aug models trained on the same corruption type are not a test of generalization; the unseen rows are.
- OOD detection uses one dataset (SVHN, loaded from a Hugging Face mirror) and one score (maximum softmax probability).
