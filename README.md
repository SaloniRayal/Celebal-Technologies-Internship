# Celebal Technologies - Data Science & Analytics Internship

This repository contains my weekly assignment modules and project submissions completed during my Data Science & Analytics internship at Celebal Technologies (June 2026).

## 🚀 Week 1 Task Overview
The core focus of this week was establishing a strong foundation in descriptive statistics, regression metrics, and exploratory analysis implementation using Python.

### Core Implementation Keypoints:
* **Descriptive Statistics:** Computed central tendency metrics (Mean, Median) and dispersion vectors (IQR, Standard Deviation) from scratch and using `Seaborn`/`Matplotlib` graphs.
* **Hypothesis Testing:** Conducted One-Sample T-Tests and Pearson Correlation analysis using `scipy.stats`.
* **Regression Metrics:** Built from-scratch validation models for Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Adjusted $R^2$ scores.
* **Distribution and Drift Testing:** Modeled Probability Density Functions (PDF/PMF), performed Kolmogorov-Smirnov (KS) testing, simulated Central Limit Theorem boundaries, and implemented Population Stability Index (PSI) equations to track data drift.

## 🛠️ Environment and Dependencies
* Google Colab / Jupyter Notebooks
* Python 3.x
* NumPy, Pandas, Matplotlib, Seaborn, SciPy, Statsmodels

  # Tesla Vehicles Delivery Forecasting Pipeline (Week 2)

An end-to-end time-series aware Machine Learning pipeline designed to forecast Tesla vehicle deliveries (2015–2025) across 4 regions and 5 vehicle models using regularized linear variants.

## 🚀 Key Pipeline Implementations
* **Exploratory Data Analysis (EDA):** Performed dimension verification, outlier analysis via boxplots, and visual distribution mapping over discrete electrical vehicle parameters.
* **Statistical Stationarity Verification:** Implemented the **Augmented Dickey-Fuller (ADF) Test** yielding a p-value of 1.31e-14, mathematically proving structural stationarity over the timeline sequence.
* **Time-Aware Feature Engineering:** Injected 1-step/2-step **Lag Features**, a 3-month **Rolling Moving Average**, and custom temporal indicators (Quarters) to capture localized multi-collinear momentum and quarter-end surges (seasonality).
* **Zero Data Leakage Boundary:** Avoided randomized row shuffling. Implemented a strict **Chronological 80/20 train-test sequence split** to respect past-to-future historical dependencies.
* **Regularized Optimization & Cross-Validation:** Trained Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1) regressions. Tuned regularized hyperparameters using `GridSearchCV` paired explicitly with a 5-fold **`TimeSeriesSplit`**.

## 📊 Final Performance Metrics
* **Best Model Variant:** Tuned Ridge Regression (alpha = 0.01)
* **Mean Absolute Error (MAE):** 106.31 units (exceptional precision over a 16,000+ volume scale)
* **R2 Variance Score:** **0.9983** (Model successfully explains 99.83% of the historical timeline variance)
