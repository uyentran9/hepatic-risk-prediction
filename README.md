---
title: "AIO 2026 - Kaggle 1 - Hepatic Risk Outcome Prediction"
layout: single
author_profile: true
--- 

<p align="center">
  <img src="{{ site.baseurl }}/images/hepatic-risk.png"
       width="960"
       alt="Hepatic Risk Outcome Prediction"
       style="border-radius: 14px;"/>
</p>


# Hepatic Risk Outcome Prediction

A reproducible tabular machine-learning pipeline for predicting three cirrhosis outcomes from demographic, clinical, laboratory, and treatment-related records.

> **Educational project:** AIO 2026 Kaggle Competition 1  
> **Task:** Multiclass probability prediction (`C`, `CL`, `D`)  
> **Primary metric:** Multiclass Log Loss  
> **Final reliable result:** OOF Log Loss **0.37010** | Public Leaderboard **0.38042**

## Project overview

The goal is to predict probabilities of three patient outcomes (sum to 1):

- `C`: alive
- `CL`: alive after liver transplant
- `D`: deceased

The competition evaluates predicted probabilities rather than only hard class labels. This makes calibration and validation design important: a confident wrong prediction is penalized much more heavily than an uncertain prediction.

The dataset contains **12,000 training rows**, **10,000 test rows**, and **18 raw predictors** composed of numerical and categorical clinical variables. The main modeling challenges are severe class imbalance, high and structured missingness, heterogeneous feature types, and substantial overlap between the rare `CL` class and the other outcomes.

## Key contributions

- Built a leakage-controlled preprocessing and validation pipeline with shared 5-fold stratified splits.
- Preserved informative missingness rather than dropping high-missing columns.
- Engineered an `in_trial` proxy from a seven-column block-missingness pattern.
- Compared Logistic Regression, XGBoost, LightGBM, and CatBoost using out-of-fold Log Loss.
- Tuned all three boosting models with a two-stage Optuna workflow: a low-cost scout round followed by a larger final search.
- Optimized ensemble weights on OOF predictions with Nelder-Mead.
- Tested seed averaging as a stability experiment and documented the disagreement between OOF and Public Leaderboard results.
- Performed class-level error analysis, showing that the rare `CL` class remains the principal source of error.

## Modeling workflow

1. **Structural audit**
   - Checked train/test column compatibility, duplicate IDs, target encoding, and unseen categories.
   - Preserved `"None"` as a valid value in `Edema_Status`.
   - Treated `Clinical_Stage` as categorical despite its numeric storage format.

2. **Exploratory data analysis**
   - Measured class imbalance: `C` 67.9%, `CL` 2.64%, `D` 29.4%.
   - Examined missingness by feature and by target class.
   - Identified a seven-column all-or-nothing missingness block.
   - Compared numerical and categorical distributions across outcomes and between train/test sets.

3. **Feature engineering and preprocessing**
   - Added `in_trial`, an inferred cohort indicator based on block completeness.
   - Added missing-value indicators for the Logistic Regression baseline.
   - Used a dedicated `__MISSING__` category for categorical missing values.
   - Applied one-hot encoding and scaling only where required by the model family.
   - Preserved clinically plausible extreme values instead of automatically removing them as outliers.

4. **Validation**
   - Shared 5-fold `StratifiedKFold` across all models.
   - Used complete OOF probability matrices for fair model comparison and blending.
   - Selected models primarily by OOF Log Loss; leaderboard scores were used only as secondary evidence.

5. **Modeling and tuning**
   - Logistic Regression baseline.
   - XGBoost, LightGBM, and CatBoost.
   - Two-round Optuna tuning per boosting model using repeated cross-validation.
   - OOF-weighted blending of tuned model probabilities.

6. **Error analysis**
   - Compared default, tuned, and final blended predictions.
   - Evaluated confusion matrices, per-class recall, class-specific Log Loss, and confidence behavior.

## Experimental results

| Experiment | Model / change | OOF Log Loss | Public LB | Decision |
|---:|---|---:|---:|---|
| 0 | Logistic Regression baseline | 0.45596 | - | Baseline |
| 1 | XGBoost, default settings | 0.37412 | 0.38372 | Kept |
| 2 | LightGBM, default settings | 0.37464 | 0.38480 | Kept |
| 3 | CatBoost, default settings | 0.38347 | 0.38953 | Kept for blending |
| 4 | Equal-weight default blend | 0.37431 | 0.38407 | Rejected |
| 5 | Optimized default blend | 0.37330 | 0.38364 | Superseded |
| 7 | Optuna-tuned XGBoost | 0.37149 | - | Kept |
| 8 | Optuna-tuned LightGBM | 0.37094 | - | Kept |
| 9 | Optuna-tuned CatBoost | 0.37952 | - | Kept |
| 10 | Optimized tuned blend | **0.37010** | **0.38042** | **Final reliable submission** |
| 11 | Tuned + seed-averaged blend | 0.36713 | 0.38067 | Inconclusive; not selected |

The seed-averaged candidate achieved a lower OOF score but a slightly worse Public Leaderboard score. Because both differences were small and the experiment added complexity without consistent external confirmation, the simpler tuned blend was retained as the final reliable submission.

## Final model

The selected solution blends Optuna-tuned boosting models using weights optimized on OOF predictions:

- **XGBoost:** 0.4308
- **LightGBM:** 0.5683
- **CatBoost:** 0.0008

The near-zero CatBoost weight indicates that the final ensemble behaves effectively as a two-model XGBoost-LightGBM blend, while retaining the complete optimization workflow for transparency.

### Final performance

- **OOF Log Loss:** 0.37010
- **Public Leaderboard Log Loss:** 0.38042
- **Submission file:** `submission_blend_optimised_v3.csv`

## Error analysis

The largest remaining weakness is the rare `CL` class.

| Class | Samples | Recall: default / tuned / blend | Mean Log Loss: default / tuned / blend |
|---|---:|---:|---:|
| C | 8,151 | 0.936 / 0.934 / 0.935 | 0.199 / 0.198 / 0.198 |
| CL | 317 | 0.158 / 0.196 / 0.192 | 2.496 / 2.486 / 2.441 |
| D | 3,532 | 0.731 / 0.735 / 0.735 | 0.588 / 0.581 / 0.581 |

Tuning slightly improved `CL` recall, but the class remained far harder than `C` or `D`. This suggests that the primary limitation is not simply hyperparameter choice; the available features provide weak separation for transplant cases, and future improvements should focus on class-specific features, weighting strategies, or better uncertainty modeling.

## Repository structure

```text
hepatic-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── hepatic_risk_outcome_prediction.ipynb
├── reports/
│   └── technical_report.pdf
├── results/
│   ├── experiment_summary.csv
│   ├── submission_blend_optimised_v3.csv   # include only if sharing is permitted
│   └── figures/
│       ├── target_distribution.png
│       ├── missingness.png
│       └── error_analysis.png
├── docs/
│   └── notebook_preview.html
└── data/
    └── README.md
```

The repository intentionally does not include an empty `src/` directory. The current project is notebook-centered; modular source files should be added only after the reusable training and evaluation functions are extracted from the notebook.

## Reproducibility

### Environment

- Python 3.13.7
- NumPy 2.3.3
- pandas 2.3.3
- scikit-learn 1.7.2
- SciPy 1.16.2
- XGBoost 3.3.0
- LightGBM 4.7.0
- CatBoost 1.2.10
- Optuna with seeded `TPESampler`

### Run instructions

1. Obtain `aio26_train.csv` and `aio26_test.csv` from the authorized competition/course source.
2. Place both files in the location expected by the notebook, or update the data-path cell.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Open `notebooks/hepatic_risk_outcome_prediction.ipynb`.
5. Run the notebook from top to bottom.
6. Step 12 performs Optuna tuning and is the slow stage (approximately 5.5 hours on the original local MacBook Pro environment).
7. The final tuned blend is created in Step 12 and validated again in the submission step.
8. Confirm that `submission_blend_optimised_v3.csv` has 10,000 rows, the required four columns, no missing values, and row-wise probabilities summing to 1.

The non-tuning sections run in approximately 2-3 minutes on the original environment. Runtime will vary by hardware.

## Limitations

- `CL` represents only 2.64% of the training data and remains difficult to separate.
- The `in_trial` feature is an inferred proxy based on missingness structure, not confirmed causal metadata.
- Blend weights were selected and evaluated on the same OOF predictions, so the small blend gain may be optimistic.
- The Optuna integer boundary check is less sensitive than the float near-boundary check.
- Seed averaging produced conflicting OOF and Public Leaderboard evidence.
- No SHAP analysis or formal feature-ablation study was completed.
- The training and tuning stages used different hardware environments.

## What I learned

The strongest lesson was that model choice alone did not determine performance. Gradient boosting produced the largest initial improvement over Logistic Regression, but careful validation, missingness handling, and systematic tuning determined whether later gains were trustworthy. The equal-weight blend was a useful failure: averaging strong models did not automatically help because CatBoost was materially weaker and received too much influence. Optimizing the weights corrected that problem.

The seed-averaging experiment was even more instructive. It improved OOF Log Loss but slightly worsened the Public Leaderboard score, showing that small validation gains can be unstable and should not be interpreted without considering variance, experimental complexity, and independent evidence. The project also showed that the rare `CL` class is fundamentally a data and feature challenge: tuning improved its recall only modestly. Future work should therefore prioritize rare-class learning, controlled feature ablations, and more rigorous uncertainty analysis rather than continuing to refine the blend.

## Data and clinical-use notice

<!--
The competition data are not included in this public repository. Obtain them only through the authorized AIO competition channel and follow the organizer's sharing rules.
--> 

This project is an educational machine-learning exercise and is **not intended for clinical decision-making or medical use**.

## Author

**Uyen Tran (Victoria Tran)**  
Individual submission, AIO 2026 Kaggle Competition 1

