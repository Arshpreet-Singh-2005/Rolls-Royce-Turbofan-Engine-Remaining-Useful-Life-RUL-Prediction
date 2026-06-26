# Rolls-Royce Turbofan Engine : Remaining Useful Life (RUL) Prediction

**Deep Learning · Bidirectional LSTM · Self-Attention · Uncertainty Estimation**

---

## Overview

This project predicts the Remaining Useful Life (RUL) of aircraft turbofan engines : the number of operational cycles remaining before failure. Accurate RUL estimates enable proactive maintenance scheduling and help prevent catastrophic in-flight engine failures.

The model is trained on the NASA C-MAPSS dataset and combines Bidirectional LSTM, Self-Attention, and Monte Carlo Dropout to produce interpretable predictions with quantified uncertainty.

---

## Key Features

| Feature | Implementation |
|---|---|
| Explainable AI (XAI) | Attention Weight Visualization : shows which time steps the model focuses on |
| Uncertainty Estimation | Monte Carlo Dropout : produces a confidence band around every RUL prediction |
| Real Sensor Integration | Gradio dashboard simulating a live sensor data stream |
| Model Optimization | Post-training quantization (float32 to float16) and pruning-aware training |

---

## Results

### Final Test Set Performance

| Metric | Value |
|---|---|
| RMSE | 16.3612 cycles |
| MAE | 12.0160 cycles |
| R-squared | 0.8333 |
| NASA Score | 696.00 (lower is better) |
| Avg Uncertainty (sigma) | 3.0114 cycles |
| Model Size (Float32) | 0.97 MB |

### Validation Set Performance (MC Dropout, 500 samples)

| Metric | Value |
|---|---|
| RMSE | 9.9479 cycles |
| MAE | 6.2290 cycles |
| R-squared | 0.9393 |
| NASA Score | 813.76 |

### Full Validation Set (20 MC passes)

| Metric | Value |
|---|---|
| RMSE | 13.6032 cycles |
| MAE | 8.9427 cycles |
| R-squared | 0.8960 |

### Training Summary

| Detail | Value |
|---|---|
| Total epochs run | 42 (early stopping at epoch 42, best weights from epoch 27) |
| Best validation MAE | 8.9178 cycles |
| Final learning rate | 0.000125 (reduced twice by ReduceLROnPlateau) |
| Training sequences | 14,184 |
| Validation sequences | 3,547 |
| Test sequences | 100 (one per engine) |
| Features per timestep | 17 |

---

## Dataset

- **Source**: NASA C-MAPSS (FD001 subset)
- **Training engines**: 100 | **Test engines**: 100
- **Engine lifetime**: mean 206.3 cycles, std 46.3, min 128, max 362
- **Features per row**: engine ID, cycle count, 3 operational settings, 21 sensor readings
- **After feature selection**: 14 sensors retained (7 dropped for near-zero variance, std < 0.01)
- **RUL labeling**: Piecewise, clipped at MAX_RUL = 125 cycles
- **Missing values**: 0 in both train and test sets

---

## Model Architecture

```
Input: (30 timesteps x 17 features)
    |
Bidirectional LSTM (128 units, return_sequences=True)
    |
Monte Carlo Dropout (p=0.3)  <-- active during inference
    |
LSTM (64 units, return_sequences=True)
    |
Monte Carlo Dropout (p=0.3)
    |
Self-Attention Layer  -->  Attention Weights (XAI)
    |
Dense(64, ReLU) --> Dense(32, ReLU) --> Dense(1) = RUL
```

### Architecture Justifications

- **Bidirectional LSTM**: Processes sequences in both directions, capturing past and future context within each sliding window for a more complete picture of the degradation trajectory.
- **Self-Attention**: Dynamically weights the importance of each of the 30 timesteps. The attention weights are exported as XAI outputs, making the model's decision process interpretable.
- **Monte Carlo Dropout**: Keeps dropout active at inference time. Running N=50 stochastic forward passes and computing the mean and standard deviation gives a probabilistic RUL estimate rather than a single point prediction.
- **Huber Loss**: Used instead of MSE during training for robustness to outlier predictions.

---

## Project Structure

```
PROJECT.ipynb              Main notebook (16 cells)
README.md                  This file
eda_plots.png              Engine lifetime distribution and sensor trend plots
feature_selection.png      Sensor variance analysis and correlation heatmap
rul_labeling.png           Standard vs piecewise RUL comparison
training_curves.png        Training loss and validation MAE over epochs
degradation_curve.png      Predicted RUL with 95% confidence bands for one engine
attention_xai.png          Attention weight bar chart and heatmap across samples
evaluation_plots.png       Predicted vs actual scatter, residual distribution, MAE by zone
test_results.png           Final test set predictions with uncertainty error bars
best_model.keras           Saved best model weights (generated during training)
```

---

## Notebook Walkthrough

**Cell 1 — Install and Import Dependencies**
Installs TensorFlow, Gradio, scikit-learn, pandas, NumPy, Matplotlib, and Seaborn. Sets random seeds (numpy: 42, tensorflow: 42) for reproducibility.

**Cell 2 — Dataset Download and Loading**
Downloads the NASA C-MAPSS FD001 files (PM_train.txt, PM_test.txt, PM_truth.txt) from a public GitHub mirror and loads them into DataFrames with named columns.

**Cell 3 — Exploratory Data Analysis**
Plots engine lifetime distribution (mean 206.3 cycles), key sensor trends for a sample engine, operational settings distribution, and a boxplot of sensor s7 across life stages. Confirms zero missing values in both splits.

**Cell 4 — Sensor Correlation Heatmap and Feature Selection**
Drops 7 constant or near-zero variance sensors (std < 0.01) and retains 14 informative sensors. Plots a correlation heatmap of retained sensors.

**Cell 5 — Piecewise RUL Target Labeling**
Applies a piecewise RUL label: true RUL values above 125 are clipped to 125, focusing model learning on the critical degradation window rather than the early healthy phase.

**Cell 6 — Normalization and Sequence Construction**
Fits a MinMaxScaler on training data only (no data leakage), transforms both splits, and builds sliding window sequences of length 30 cycles. Produces 14,184 train / 3,547 validation / 100 test sequences, each with 17 features per timestep.

**Cell 7 — Model Architecture**
Defines the SelfAttention layer and the full BiLSTM model. The model outputs both the RUL prediction and the attention weights.

**Cell 8 — Model Training**
Trains with Huber loss for up to 80 epochs. EarlyStopping (patience 15) stops at epoch 42 and restores best weights from epoch 27. ReduceLROnPlateau fires twice, reaching a final learning rate of 0.000125. Best validation MAE: 8.9178 cycles.

**Cell 9 — Training Curves Visualization**
Plots training Huber loss and validation MAE over all 42 epochs, confirming healthy convergence without overfitting.

**Cell 10 — Monte Carlo Dropout Inference**
Runs 50 stochastic forward passes on 500 validation samples. Reports RMSE 9.9479, MAE 6.2290, R-squared 0.9393, NASA Score 813.76.

**Cell 11 — Degradation Curve with Uncertainty Bands**
Plots the predicted RUL at every timestep for one engine alongside the true RUL and a 95% confidence band (mean +/- 2 standard deviations). Highlights the critical zone below RUL = 30.

**Cell 12 — Attention Visualization (XAI)**
Shows a bar chart of attention weights for one sample sequence and a heatmap of attention patterns across 40 samples, revealing which timesteps the model focuses on most.

**Cell 13 — Predicted vs Actual Scatter and Residual Analysis**
Three plots: predicted vs actual scatter with R-squared annotation, residual histogram, and MAE broken down by RUL zone (Critical, Warning, Monitor, Healthy). Full validation RMSE 13.6032, MAE 8.9427, R-squared 0.8960.

**Cell 14 — Gradio Dashboard**
Launches an interactive web dashboard with sliders for six key sensors (s2, s3, s4, s7, s11, s12). Runs 30 MC Dropout passes per prediction and displays the estimated RUL, 95% confidence interval, health status recommendation, and an attention visualization.

**Cell 15 — Test Set Evaluation**
Evaluates on all 100 held-out test engines using NASA's asymmetric scoring function. Final results: RMSE 16.3612, MAE 12.0160, R-squared 0.8333, NASA Score 696.00, Avg Uncertainty 3.0114 cycles.

**Cell 16 — Summary and Deployment Plan**
Prints the final project summary and outlines a multi-stage deployment strategy.

---

## Setup and Usage

### Requirements

```
Python >= 3.9
tensorflow >= 2.12
gradio
scikit-learn
pandas
numpy
matplotlib
seaborn
```

### Installation

```bash
pip install tensorflow gradio scikit-learn pandas numpy matplotlib seaborn
```

### Running the Notebook

1. Clone or download this repository.
2. Open `EST_PROJECT.ipynb` in Jupyter Notebook, JupyterLab, or upload to Google Colab (recommended — uses T4 GPU).
3. Run cells sequentially from Cell 1 to Cell 16.
4. The dataset is downloaded automatically in Cell 2.
5. The Gradio dashboard (Cell 14) will generate a public share link accessible in any browser.

---

## Deployment Plan

| Stage | Technology | Purpose |
|---|---|---|
| Edge | TFLite Float16 (~0.48 MB) | On-board avionics, low-power real-time inference |
| Ground | FastAPI + Docker | Fleet-level prediction API backend |
| Dashboard | Gradio or Streamlit | Engineer-facing UI with XAI and uncertainty display |
| Alerts | WebSocket push | Real-time notifications when RUL crosses critical threshold |

---

## NASA Scoring Function

The evaluation uses an asymmetric penalty designed for safety-critical RUL prediction:

- Late predictions (predicted RUL higher than true) are penalized more heavily via exp(d/10) : underestimating engine wear risks in-flight failure.
- Early predictions (predicted RUL lower than true) are penalized less via exp(-d/13) : a slight overestimate leads only to premature maintenance.

The model achieved a NASA Score of 696.00 on the test set (lower is better).

---

## Ethical Considerations

- False negatives (high predicted RUL when true RUL is low) carry a greater safety penalty and are explicitly discouraged through the asymmetric NASA scoring function.
- Monte Carlo Dropout ensures uncertainty is always surfaced alongside predictions, preventing blind reliance on point estimates. Average uncertainty on the test set is 3.0114 cycles.
- Attention weights make the model's reasoning transparent, eliminating black-box decision-making in a safety-critical context.

---

## Dataset Reference

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

---

## Author

**Arshpreet Singh**
