---
layout: single
title: "Detailed Project Pipeline — Hepatic Risk Outcome Prediction"
permalink: /pipeline/
author_profile: true
toc: true
toc_label: "Project pipeline"
toc_icon: "cogs"
toc_sticky: true
---

> **AIO 2026 Kaggle Competition 1**  
> **Author:** Uyen Tran (Victoria Tran)  
> **Task:** Predict three possible hepatic outcomes from structured clinical records  
> **Evaluation metric:** Multiclass Log Loss  
> **Final reliable submission:** `submission_blend_optimised_v3.csv`  
> **Final reliable scores:** OOF Log Loss **0.37010** · Public Leaderboard **0.38042**  
> **Documentation basis:** Checked against the final executed notebook export `Kaggle1_Uyen8.html` and the saved experiment evidence.  
> **Scope:** This is a detailed explanatory companion to the notebook, not a replacement for the executable code.

---

## Overview

The notebook contains the executable code. The technical report provides a concise, competition-formatted summary. This file provides the detailed narrative that connects the problem, data, code, outputs, decisions, failed experiments, and final conclusion.

A recurring pattern is used throughout:

```text
Question
  ↓
Observation from data or experiment
  ↓
Interpretation
  ↓
Modeling decision
  ↓
Measured result
  ↓
Keep, reject, or investigate further
```

This matters because a strong machine-learning project is not simply a sequence of algorithms. It is a sequence of decisions supported by evidence.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Problem definition](#2-problem-definition)
3. [Dataset overview](#3-dataset-overview)
4. [Design principles](#4-design-principles)
5. [End-to-end pipeline](#5-end-to-end-pipeline)
6. [Step 1 — Imports and configuration](#6-step-1--imports-and-configuration)
7. [Step 2 — Data loading](#7-step-2--data-loading)
8. [Step 3 — Structural data audit](#8-step-3--structural-data-audit)
9. [Step 4 — Exploratory data analysis](#9-step-4--exploratory-data-analysis)
10. [Step 5 — Feature matrix construction](#10-step-5--feature-matrix-construction)
11. [Step 6 — Validation strategy](#11-step-6--validation-strategy)
12. [Step 7 — Logistic Regression baseline](#12-step-7--logistic-regression-baseline)
13. [Step 8 — XGBoost](#13-step-8--xgboost)
14. [Step 9 — LightGBM](#14-step-9--lightgbm)
15. [Step 10 — CatBoost](#15-step-10--catboost)
16. [Step 11 — Model comparison](#16-step-11--model-comparison)
17. [Step 12 — Optuna hyperparameter tuning](#17-step-12--optuna-hyperparameter-tuning)
18. [Step 13 — Seed averaging](#18-step-13--seed-averaging)
19. [Step 14 — Default-model blending](#19-step-14--default-model-blending)
20. [Step 15 — Error analysis](#20-step-15--error-analysis)
21. [Step 16 — Submission generation](#21-step-16--submission-generation)
22. [Step 17 — Experiment evidence and reproducibility](#22-step-17--experiment-evidence-and-reproducibility)
23. [Final model and experimental results](#23-final-model-and-experimental-results)
24. [What worked, what failed, and what was learned](#24-what-worked-what-failed-and-what-was-learned)
25. [Limitations](#25-limitations)
26. [Future work](#26-future-work)
27. [Reproduction guide](#27-reproduction-guide)
28. [Glossary](#28-glossary)
29. [Verified visual and output audit](#29-verified-visual-and-output-audit)
30. [Notebook cell map](#30-notebook-cell-map)

---

# 1. Executive summary

The goal of this project is to predict a patient's hepatic outcome at a follow-up point using demographic information, clinical findings, laboratory measurements, and treatment-related records.

For every patient in the hidden test set, the model must produce three probabilities:

- `Status_C`: probability that the patient is alive;
- `Status_CL`: probability that the patient is alive after liver transplantation;
- `Status_D`: probability that the patient is deceased.

These probabilities must be nonnegative and sum to one. The competition is therefore not asking only, “Which class is most likely?” It is also asking, “How confident should the model be?”

The main difficulties were:

1. **Severe class imbalance.** Only 317 of 12,000 training rows belong to `CL`, approximately 2.64% of the data.
2. **Substantial missing data.** Eleven predictor columns contain missing values, and two are missing in more than half of the rows.
3. **Structured missingness.** Seven columns are usually complete together or missing together, indicating a data-collection pattern rather than independent random omissions.
4. **Mixed data types.** The dataset contains numerical measurements, mostly nominal categorical variables, and an ordinal-like clinical-stage variable that require different preprocessing choices.
5. **Probability quality.** The Log Loss metric penalizes confident mistakes, so reliable probability estimates matter more than raw accuracy alone.

The project followed an evidence-driven workflow:

```text
Load data carefully
→ audit structure
→ analyze distributions and missingness
→ construct leakage-safe features
→ create shared stratified folds
→ establish a linear baseline
→ train three boosting models
→ tune them with a two-round Optuna search
→ blend their OOF probabilities
→ test stability through seed averaging
→ analyze class-specific errors
→ validate and export submission files
→ save reproducibility evidence
```

The strongest reliable model was an OOF-weighted blend of Optuna-tuned XGBoost, LightGBM, and CatBoost:

| Component | Blend weight |
|---|---:|
| XGBoost | 0.4308 |
| LightGBM | 0.5683 |
| CatBoost | 0.0008 |

The final scores were:

- **OOF Log Loss:** 0.37010
- **Public Leaderboard Log Loss:** 0.38042

A later seed-averaging experiment produced a better OOF score of 0.36713 but a slightly worse Public Leaderboard score of 0.38067. Because the evidence disagreed and the gain was small relative to observed variability, seed averaging was treated as inconclusive rather than a confirmed improvement. The simpler tuned blend was retained as the final reliable submission.

---

# 2. Problem definition

## 2.1 What is the model predicting?

Each training row represents one patient. The model uses available patient information to estimate the probability of three possible outcomes at a follow-up point:

| Label | Meaning |
|---|---|
| `C` | Alive at the follow-up point |
| `CL` | Alive after liver transplantation |
| `D` | Deceased |

This is a **multiclass classification** problem because the target belongs to one of three discrete categories.

It is not a regression problem. Regression predicts a continuous number, such as hospital cost or survival time. This project predicts a categorical outcome.

## 2.2 Why probabilities are required

A normal classifier may output only one label, such as `D`. This competition instead requires a probability distribution:

```text
Status_C  = 0.20
Status_CL = 0.05
Status_D  = 0.75
```

The probabilities communicate uncertainty. The model is saying that `D` is most likely, but it is not claiming absolute certainty.

This distinction is important in medical and biomedical settings, where uncertainty is often meaningful. Two patients may receive the same predicted class but very different confidence levels.

## 2.3 What does Log Loss measure?

Multiclass Log Loss examines the probability assigned to the true class. Lower is better.

Consider a patient whose true outcome is `D`:

| Prediction | Interpretation |
|---|---|
| `C=0.30, CL=0.20, D=0.50` | Correct top class, moderate confidence |
| `C=0.95, CL=0.04, D=0.01` | Confidently wrong |

The second prediction receives a much larger penalty because it assigns almost no probability to the correct answer.

This leads to an important design principle:

> A model should not only classify correctly. It should also express confidence responsibly.

For that reason, this project compares models with OOF Log Loss rather than overall accuracy.

---

# 3. Dataset overview

## 3.1 Dataset sizes

| Dataset | Rows | Columns | Includes target? |
|---|---:|---:|---|
| Training set | 12,000 | 20 | Yes |
| Test set | 10,000 | 19 | No |

The training set contains:

- one ID column;
- 18 raw predictors;
- the target column `Status`.

The test set contains the same ID and predictor columns but does not contain `Status`. The model must estimate the missing outcomes.

## 3.2 Feature groups

The predictors cover several kinds of information:

### Demographic and timing information

Examples include patient age and follow-up duration.

### Clinical findings

Examples include ascites, liver enlargement, spider angioma, edema status, and clinical stage.

### Laboratory measurements

Examples include bilirubin, albumin, cholesterol, copper, alkaline phosphatase, AST, triglycerides, platelet count, and prothrombin time.

### Treatment-related information

The dataset also includes treatment assignment and a recurring seven-column completeness pattern associated with the recorded trial block.

## 3.3 Complete raw-feature dictionary

The final code starts from **18 raw predictors**. The table below records how each variable enters the project. The dataset materials do not provide enough information to claim measurement units or causal clinical meaning beyond the column names, so this documentation does not invent them.

| Raw predictor | Data role in the notebook | Modeling treatment | Why this treatment was used |
|---|---|---|---|
| `Followup_Days` | Numerical timing variable | Numerical | A continuous value; retained without scaling for trees and scaled inside the Logistic Regression fold pipeline. |
| `Treatment_Assignment` | Categorical treatment record | Categorical | Its levels are labels rather than quantities; one-hot encoded for Logistic Regression and handled natively by boosting libraries. |
| `Patient_Age_Days` | Numerical demographic variable | Numerical | Preserved as a continuous value. |
| `Patient_Sex` | Categorical demographic variable | Categorical | No numerical ordering should be imposed. |
| `Ascites_Indicator` | Categorical clinical finding | Categorical | A clinical state represented by categories; also participates in the seven-column completeness pattern. |
| `Liver_Enlargement` | Categorical clinical finding | Categorical | Retained because outcome composition changes visibly across its levels. |
| `Spider_Angioma` | Categorical clinical finding | Categorical | Retained because its category levels show different outcome compositions. |
| `Edema_Status` | Categorical clinical finding | Categorical | The literal value `None` is a valid category, so custom CSV parsing is required to prevent it from becoming `NaN`. |
| `Bilirubin_Level` | Numerical laboratory variable | Numerical | Strongly skewed and visibly different across outcome classes; trees can use thresholds without scaling. |
| `Cholesterol_Level` | Numerical laboratory variable | Numerical with missingness preserved | More than half of its values are missing, and missingness varies across target classes. |
| `Albumin_Level` | Numerical laboratory variable | Numerical | Retained because its distribution differs across classes. |
| `Copper_Level` | Numerical laboratory variable | Numerical with missingness preserved | Part of the structured seven-column completeness block. |
| `Alkaline_Phosphatase` | Numerical laboratory variable | Numerical with missingness preserved | Part of the structured block and strongly right-skewed. |
| `AST_Level` | Numerical laboratory variable | Numerical with missingness preserved | Part of the structured block. |
| `Triglyceride_Level` | Numerical laboratory variable | Numerical with missingness preserved | Missing in about 55.8% of training rows; not dropped because missingness itself is informative. |
| `Platelet_Count` | Numerical laboratory variable | Numerical with missingness preserved | Mostly complete but still receives fold-safe imputation in Logistic Regression. |
| `Prothrombin_Time` | Numerical laboratory variable | Numerical with missingness preserved | Almost complete; retained as a continuous predictor. |
| `Clinical_Stage` | Ordinal-like clinical category stored numerically | Treated as categorical in this pipeline | The values have an apparent progression, but the code avoids assuming that equal numeric gaps imply equal risk changes. |

The code then adds one engineered predictor:

| Engineered predictor | Definition | Interpretation |
|---|---|---|
| `in_trial` | `1` when all seven trial-block fields are present; `0` otherwise | A **block-completeness proxy**. It captures a real data-collection boundary, but it is not proof of actual trial participation for every row. |

After feature construction, the shared matrix has **19 predictors**: 12 numerical variables including `in_trial`, and 7 categorical variables.

## 3.4 Target distribution

| Class | Rows | Percentage |
|---|---:|---:|
| `C` | 8,151 | 67.92% |
| `CL` | 317 | 2.64% |
| `D` | 3,532 | 29.43% |

The `CL` class is extremely rare. A model can achieve high overall accuracy while almost never identifying `CL` correctly. This is why class-specific error analysis is necessary.

## 3.5 Missing data

Eleven predictor columns contain missing values. Two laboratory variables are missing in more than 55% of the training rows. A group of seven columns is missing in approximately 43% of rows, usually together.

The project does not automatically interpret missing data as a defect to remove. It asks two questions first:

1. Does the missingness pattern differ across outcome classes?
2. Do multiple variables disappear together in a meaningful structural pattern?

The answer to both questions was yes, which made missingness part of the modeling signal.

---

# 4. Design principles

The pipeline was built around six principles.

## 4.1 Prevent data leakage

Data leakage occurs when information from validation or test data influences model training. Leakage makes validation results look better than the model's real generalization ability.

For example, computing the median of a numerical column on all 12,000 training rows before cross-validation would allow each validation fold to influence the imputation value used to transform itself.

To avoid this:

- only structural, non-learned transformations are applied before cross-validation;
- learned preprocessing for Logistic Regression is fit inside each training fold;
- model selection is based on OOF predictions produced for rows the model did not train on.

## 4.2 Use the same folds for fair comparison

All default models are trained and evaluated on the same five stratified folds.

This controls an important source of noise. If XGBoost and LightGBM used different splits, a score difference could be caused by one model receiving an easier validation set rather than by the algorithm itself.

Shared folds also align the OOF rows exactly, which is required for valid blending.

## 4.3 Optimize the competition metric directly

The competition uses Log Loss, so every major comparison uses Log Loss.

Accuracy, confusion matrices, and recall are useful diagnostic tools, but they do not replace the official probability-based metric.

## 4.4 Make EDA actionable

Each major EDA question ends in a modeling decision. Examples:

| Observation | Decision |
|---|---|
| `CL` is only 2.64% of training data | Use stratified folds |
| Missingness differs across classes | Preserve missingness indicators |
| Seven columns are missing together | Add a block-completeness proxy |
| Most categorical variables are nominal; `Clinical_Stage` is ordinal-like | Avoid false ordering; document the categorical/ordinal trade-off |
| Numerical scales differ greatly | Scale for Logistic Regression only |

## 4.5 Prefer OOF evidence to leaderboard chasing

The public leaderboard evaluates only part of the hidden test labels. It is therefore a noisy and incomplete estimate of final performance.

The primary model-selection signal is OOF Log Loss. Public leaderboard results are treated as secondary evidence.

This principle does not mean ignoring a contradiction. When OOF and the leaderboard disagree, the correct response is to investigate uncertainty rather than blindly trust either one.

## 4.6 Document failures, not only successes

Several experiments were rejected:

- an arbitrary ratio feature did not improve validation;
- equal-weight blending underperformed the best individual model;
- seed averaging improved OOF but slightly worsened the Public Leaderboard.

These failures are included because they reveal how the pipeline was evaluated and corrected.

---

# 5. End-to-end pipeline

The final notebook is organized into 17 steps:

```text
Step 1   Imports and configuration
Step 2   Data loading
Step 3   Structural data audit
Step 4   Exploratory data analysis
Step 5   Feature matrix construction
Step 6   Shared stratified validation framework
Step 7   Logistic Regression baseline
Step 8   XGBoost
Step 9   LightGBM
Step 10  CatBoost
Step 11  Default-model comparison
Step 12  Two-round Optuna tuning and tuned blend
Step 13  Multi-seed averaging experiment
Step 14  Default-model equal and optimized blends
Step 15  Error analysis before and after tuning
Step 16  Submission file generation
Step 17  Experiment evidence and reproducibility artifacts
```

The order in the notebook reflects the project's development history. Step 14 preserves the original default-model blending experiment for comparison, while the final tuned blend is constructed in Step 12 after hyperparameter optimization.

## 5.1 What happens when the notebook is run top to bottom?

A Jupyter notebook uses a Python **kernel**, which is a live Python process that keeps variables in memory.

```text
Cell 1 runs
→ variables created in Cell 1 remain in RAM
→ Cell 2 can use them
→ later cells add models, predictions, scores, and files
```

Three kinds of lines behave differently:

```python
# A comment
```

Python ignores comments; they explain the intention of the code.

```python
def run_cv(...):
    ...
```

This defines and stores a reusable procedure. The code inside the function does not execute until the function is called.

```python
run_cv("XGBoost", train_xgb)
```

This actually runs the procedure, trains the fold models, creates predictions, calculates scores, and writes the result into `RESULTS`.

This distinction matters when reading the HTML export. A long function cell may only define the method, while the output shown below the cell comes from the function call at the end.

## 5.2 Core data objects and their shapes

The following objects are the backbone of the notebook:

| Object | Meaning | Shape |
|---|---|---:|
| `train` | Original training DataFrame, including `id` and `Status` | 12,000 × 20 |
| `test` | Original hidden-label test DataFrame, including `id` | 10,000 × 19 |
| `X` | Training predictors after removing `id`/`Status` and adding `in_trial` | 12,000 × 19 |
| `X_test` | Test predictors after removing `id` and adding `in_trial` | 10,000 × 19 |
| `y` | Encoded training labels: `C=0`, `CL=1`, `D=2` | 12,000 |
| `X_tr` | Four folds used to fit one fold model | 9,600 × 19 |
| `X_va` | One fold withheld for validation | 2,400 × 19 |
| `y_tr` | True labels for `X_tr` | 9,600 |
| `y_va` | True labels for `X_va` | 2,400 |
| `proba_va` | Probabilities predicted for the current validation fold | 2,400 × 3 |
| `proba_te` | Probabilities predicted for all test rows by the current fold model | 10,000 × 3 |
| `oof` | Validation probabilities restored to all original training-row positions | 12,000 × 3 |
| `test_pred` | Average of the five fold models' test probabilities | 10,000 × 3 |

The labels and probability arrays must not be confused:

```text
y_tr / y_va
= true class IDs such as [0, 0, 2, 1, ...]

proba_va / proba_te
= three probabilities per row, such as [0.80, 0.03, 0.17]
```

## 5.3 The journey of one training row

Suppose Patient A belongs to validation fold 3:

```text
Fold 1: Patient A is used for training
Fold 2: Patient A is used for training
Fold 3: Patient A is withheld and predicted
Fold 4: Patient A is used for training
Fold 5: Patient A is used for training
```

Only the fold-3 prediction is stored as Patient A's OOF prediction. The model producing that prediction never trained on Patient A.

That is why OOF predictions are appropriate for honest local evaluation.

## 5.4 The journey of one test row

A test row has no known `Status`, so it is never used for fitting. Each of the five fold models predicts it:

```text
Fold model 1 → probability vector
Fold model 2 → probability vector
Fold model 3 → probability vector
Fold model 4 → probability vector
Fold model 5 → probability vector
                     ↓
              row-wise average
                     ↓
         one final 3-class probability vector
```

For example:

| Fold model | P(C) | P(CL) | P(D) |
|---|---:|---:|---:|
| 1 | 0.80 | 0.03 | 0.17 |
| 2 | 0.75 | 0.04 | 0.21 |
| 3 | 0.82 | 0.02 | 0.16 |
| 4 | 0.77 | 0.03 | 0.20 |
| 5 | 0.81 | 0.02 | 0.17 |
| **Average** | **0.790** | **0.028** | **0.182** |

## 5.5 Complete information flow

```text
train.csv
  ↓
train DataFrame
  ↓ remove id and Status; add in_trial
X (12,000 × 19) + y (12,000)
  ↓ shared 5-fold StratifiedKFold
five fitted models per model family
  ↓
OOF matrix (12,000 × 3) ──→ local Log Loss, diagnostics, blend fitting
test matrix (10,000 × 3) ─→ submission generation

test.csv
  ↓ remove id; add in_trial
X_test (10,000 × 19)
  ↓ predicted by every fold model
average fold predictions
  ↓
attach original test IDs
  ↓
submission CSV
```

---

# 6. Step 1 — Imports and configuration

## Purpose

Step 1 prepares the software environment and defines the project-wide constants used later.

It does not train a model. Its role is to make later work consistent and reproducible.

## Main tools

| Tool | Role |
|---|---|
| NumPy | Probability arrays, averaging, blending, saved OOF files |
| pandas | Reading, inspecting, transforming, and exporting tabular data |
| Matplotlib / seaborn | Visual analysis |
| scikit-learn | Cross-validation, Logistic Regression, preprocessing, metrics |
| XGBoost | Gradient-boosted trees |
| LightGBM | Leaf-wise gradient-boosted trees |
| CatBoost | Boosting with native categorical handling |
| Optuna | Hyperparameter optimization |
| SciPy `minimize` | Blend-weight optimization |

## Central configuration

The notebook defines constants such as:

```text
TARGET      = "Status"
ID_COL      = "id"
CLASSES     = ["C", "CL", "D"]
N_FOLDS     = 5
SEED        = 42
```

The order of `CLASSES` is particularly important. Every probability matrix is interpreted as:

```text
column 0 → P(C)
column 1 → P(CL)
column 2 → P(D)
```

If the order were changed accidentally, the code might still run while assigning probabilities to the wrong submission columns. The structural audit later checks this alignment.

## The `RESULTS` registry

The notebook initializes a shared dictionary called `RESULTS`.

Each model writes a standardized record containing fields such as:

```text
cv_score
fold_scores
fold_std
oof
test_pred
runtime_s
```

This registry is the central experiment ledger. Later sections can compare models, generate submissions, perform error analysis, and export evidence without retraining everything.

## Why versions are recorded

Machine-learning libraries can change implementation details between releases. Re-running the same notebook under different versions can produce slightly different results.

The project therefore records the exact Python and package versions used for the final pipeline.

## Exact output from the final HTML run

```text
Libraries imported successfully.
  numpy      2.3.3
  pandas     2.3.3
  xgboost    3.3.0
  lightgbm   4.7.0
  catboost   1.2.10

Configuration set.
  Folds : 5  |  Seed: 42  |  Classes: ['C', 'CL', 'D']
```

This output confirms two different things:

1. the required libraries imported successfully under the recorded versions;
2. the global fold count, random seed, and class ordering match the intended configuration.

The `%pip install` cell also prints a generic message saying that the kernel *may* need to restart. That message is not evidence of failure. The relevant confirmation is that the imports succeed and the version lines print afterward.

## Small code-cleanup observations

The final HTML contains two consecutive `RESULTS = {}` lines. The second assignment is harmless because the dictionary is still empty at that moment, but one line should be removed for clarity.

The code comment above `TRIAL_BLOCK` should also avoid stating that incomplete rows are definitely non-enrolled trial patients. The data show a strong completeness pattern, but actual trial participation is not independently verified. The safer wording is:

```text
TRIAL_BLOCK represents a seven-column completeness pattern that may reflect
a trial or cohort boundary; in_trial is used only as an inferred proxy.
```

### Step 1 outcome

At the end of Step 1, the environment is ready, the class order is explicit, reproducibility controls are set, and the shared experiment registry exists.

---

# 7. Step 2 — Data loading

## Purpose

The goal is not merely to open the CSV files. The goal is to read them without silently changing valid clinical categories.

## The `Edema_Status = "None"` issue

The string `"None"` appears as a genuine category in `Edema_Status`, meaning no edema is present. It is not a missing value.

Some CSV readers may interpret text such as `None`, `NA`, or `null` as missing by default. If that happened here, a valid clinical category representing most patients would be erased and the missing-value analysis would become severely distorted.

The notebook therefore uses custom NA settings:

```python
read_opts = dict(
    keep_default_na=False,
    na_values=[""]
)
```

This means:

- preserve the literal string `None`;
- treat genuinely empty cells as missing.

## Loaded objects

After loading:

```text
train → 12,000 rows × 20 columns
test  → 10,000 rows × 19 columns
```

The original CSV files are not modified. pandas creates DataFrames in memory.

## Initial observations

The first rows already reveal a recurring pattern: several trial-related fields and laboratory values are absent together, while other measurements remain available.

This early observation motivates a formal block-missingness analysis in Step 4.

### Step 2 outcome

The data are loaded with valid categorical meanings preserved. The shapes match expectations, and the notebook can now audit the dataset safely.

---

# 8. Step 3 — Structural data audit

## Purpose

A structural audit is a set of safety checks designed to catch silent problems before modeling begins.

A silent problem is dangerous because the code may run without an obvious error while producing invalid predictions.

## Audit 1 — Target classes

The notebook verifies that the observed target values are exactly:

```text
C, CL, D
```

It also confirms the intended submission columns:

```text
Status_C, Status_CL, Status_D
```

## Audit 2 — Train/test feature alignment

The training and test sets must contain the same 18 predictors.

The only expected difference is:

- training data contain `Status`;
- test data do not.

The audit confirms that there are no train-only or test-only predictor columns.

## Audit 3 — Duplicate IDs

The notebook verifies that no patient ID is duplicated in either dataset.

Duplicate IDs could indicate repeated records, data corruption, or submission alignment problems.

## Audit 4 — Target encoding

The text labels are mapped to integers:

```text
C  → 0
CL → 1
D  → 2
```

The audit verifies that no label fails to map. An unexpected label would become missing during mapping and later cause incorrect training behavior.

## Audit 5 — Unseen categorical levels

For every categorical predictor, the notebook checks whether the test set contains a category never observed in training.

No unseen test categories were found. The Logistic Regression encoder still uses `handle_unknown="ignore"` as a defensive safeguard.

## Audit result

All five checks pass:

- expected target classes are present;
- train and test predictors align;
- IDs are unique;
- target encoding is complete;
- test categories are represented in training.

### Why a clean audit is still valuable

The value of an audit is not that it finds an error every time. Its value is that it converts assumptions into verified conditions.

### Step 3 outcome

The pipeline proceeds with evidence that the dataset structure is internally consistent.

---

# 9. Step 4 — Exploratory data analysis

Exploratory data analysis, or EDA, asks questions before choosing a modeling strategy.

This project focuses on seven practical questions.

## 9.1 Target distribution

### Question

Are the outcome classes balanced?

### Observation

```text
C  = 8,151 rows (67.92%)
CL =   317 rows ( 2.64%)
D  = 3,532 rows (29.43%)
```

### Interpretation

`CL` is rare enough that a normal random split could produce unstable validation subsets. A model may also minimize average loss mainly by learning `C` and `D`, while performing poorly on `CL`.

### Decision

Use `StratifiedKFold` so every validation fold preserves approximately the same class proportions.

---

## 9.2 Overall missingness

### Question

Which predictors contain missing data, and how much?

### Observation

Eleven predictors contain missing values:

| Feature | Missing rate |
|---|---:|
| `Triglyceride_Level` | 55.8% |
| `Cholesterol_Level` | 55.5% |
| `Copper_Level` | 43.8% |
| `AST_Level` | 43.2% |
| `Alkaline_Phosphatase` | 43.2% |
| `Treatment_Assignment` | 43.1% |
| `Spider_Angioma` | 43.1% |
| `Liver_Enlargement` | 43.1% |
| `Ascites_Indicator` | 43.1% |
| `Platelet_Count` | 3.7% |
| `Prothrombin_Time` | 0.1% |

Two laboratory columns are missing in more than half of the rows. More importantly, seven columns cluster tightly around 43%, which suggests that they are governed by a common collection process.

### Interpretation

The similar rates among seven different columns are unlikely to be a coincidence. They suggest a shared data-collection mechanism.

### Decision

Do not drop high-missing columns automatically. First investigate whether missingness carries information.

---

## 9.3 Missingness by target class

### Question

Does the probability of a value being missing differ among `C`, `CL`, and `D`?

### Observation

Missing rates differ by outcome:

| Feature | C | CL | D | Max–min spread |
|---|---:|---:|---:|---:|
| `Cholesterol_Level` | 55.1% | 46.1% | 57.2% | 11.1 pp |
| `Triglyceride_Level` | 55.4% | 46.7% | 57.6% | 10.9 pp |
| `Copper_Level` | 43.3% | 38.8% | 45.5% | 6.7 pp |
| `Ascites_Indicator` | 42.7% | 38.5% | 44.4% | 5.9 pp |
| `AST_Level` | 42.8% | 38.8% | 44.5% | 5.7 pp |
| `Alkaline_Phosphatase` | 42.8% | 38.8% | 44.5% | 5.7 pp |
| `Spider_Angioma` | 42.7% | 38.8% | 44.5% | 5.7 pp |
| `Treatment_Assignment` | 42.8% | 38.8% | 44.4% | 5.6 pp |
| `Liver_Enlargement` | 42.7% | 38.8% | 44.3% | 5.5 pp |
| `Platelet_Count` | 3.5% | 0.6% | 4.4% | 3.8 pp |
| `Prothrombin_Time` | 0.1% | 0.0% | 0.2% | 0.2 pp |

A recurring pattern is:

- `D` often has the highest missingness;
- `CL` often has the lowest;
- `C` lies between them.

“pp” means percentage points. For example, 57.2% minus 46.1% equals an 11.1-percentage-point spread.

### Interpretation

Missingness is not merely absence of information. Whether a test was recorded may indirectly reflect clinical workflow, follow-up intensity, patient subgroup, or disease trajectory.

The exact causal reason cannot be confirmed from the provided data, so the project treats missingness as predictive structure rather than making a clinical causal claim.

### Decision

- Preserve missingness through indicator variables for Logistic Regression.
- Allow boosting models to learn native missing-value directions where supported.
- Represent missing categorical values explicitly rather than forcing them into an existing category.

---

## 9.4 Seven-column block missingness

### Question

Are the seven trial-related columns missing independently, or together?

### Observation

For approximately 99% of rows, the block is either:

- fully complete; or
- fully missing.

Exact breakdown from the notebook:

| Number of missing fields in the seven-column block | Rows | Percentage |
|---:|---:|---:|
| 0 | 6,737 | 56.1% |
| 1 | 79 | 0.7% |
| 2 | 3 | 0.0% |
| 3 | 1 | 0.0% |
| 4 | 1 | 0.0% |
| 5 | 2 | 0.0% |
| 6 | 30 | 0.2% |
| 7 | 5,147 | 42.9% |

The two extreme states—completely observed and completely missing—cover about 99% of all rows.

### Interpretation

This is a strong structural pattern. It resembles a cohort or data-collection boundary rather than seven independent missing events.

### Decision

Create a binary proxy called `in_trial`:

```text
in_trial = 1 when all seven block fields are present
in_trial = 0 otherwise
```

### Important caution

The name is an inferred modeling label. The dataset does not independently prove actual trial participation for every row. In this documentation, `in_trial` should be understood as a **trial-block completeness proxy**, not confirmed causal metadata.

---

## 9.5 Numerical distributions by class

### Question

Do numerical predictors show different distributions across outcomes?

### Observation

Several liver-related variables shift toward more severe values for class `D`. Many features are strongly right-skewed and have very different numerical scales.

The `CL` curves are often noisy and overlap with both `C` and `D`, partly because only 317 examples are available.

### Interpretation

- Individual variables contain predictive signal.
- Relationships are unlikely to be purely linear.
- Feature interactions may be important.
- Scaling is necessary for a linear model but unnecessary for tree splits.

### Decision

- Scale numerical variables for Logistic Regression.
- Do not scale them for tree-based models.
- Test nonlinear boosting models capable of learning interactions.

---

## 9.6 Categorical features and outcomes

### Question

Do categorical levels have different outcome compositions?

### Observation

Several categorical levels show meaningful shifts in the proportions of `C`, `CL`, and `D`. Clinical findings such as persistent edema or present ascites are associated with different outcome distributions.

### Interpretation

The categorical variables contain signal and should be retained.

Most are **nominal**, meaning their values do not have a valid numerical ranking. Encoding treatment categories as 0, 1, and 2 would create an artificial order.

`Clinical_Stage` is the exception: stages 1–4 have a natural order. In this pipeline it is still treated as categorical so the model is not forced to assume that the effect is linear and equally spaced. The trade-off is that the explicit order `1 < 2 < 3 < 4` is not directly encoded; this remains an ablation experiment for future work.

### Decision

- Use one-hot encoding for Logistic Regression.
- Use the boosting libraries' supported categorical mechanisms for XGBoost, LightGBM, and CatBoost.
- Treat `Clinical_Stage` as categorical in the current pipeline despite its numeric storage format.

---

## 9.7 Train/test distribution comparison

### Question

Are the training and test datasets drawn from visibly different distributions?

### Observation

Selected numerical distributions and summary statistics overlap closely:

| Feature | Train mean | Test mean | Train SD | Test SD | Train missing | Test missing |
|---|---:|---:|---:|---:|---:|---:|
| `Followup_Days` | 1978.274 | 1964.716 | 1380.381 | 1243.237 | 0.0% | 0.0% |
| `Patient_Age_Days` | 19311.575 | 19305.011 | 3942.175 | 3657.610 | 0.0% | 0.0% |
| `Bilirubin_Level` | 1.810 | 1.853 | 2.605 | 2.733 | 0.0% | 0.0% |
| `Cholesterol_Level` | 329.262 | 330.536 | 183.345 | 189.082 | 55.5% | 56.2% |
| `Albumin_Level` | 3.524 | 3.527 | 0.373 | 0.375 | 0.0% | 0.0% |
| `Copper_Level` | 74.769 | 74.914 | 73.227 | 76.435 | 43.8% | 44.5% |

The means, spreads, and missing rates are close for these inspected features.

### Interpretation

No severe univariate distribution shift is evident in the inspected variables.

This does not prove that the two datasets are identical. Categorical or multivariate shift may still exist.

### Decision

Use cross-validation as the main model-selection criterion and treat the public leaderboard as a secondary, noisier signal.

---

## 9.8 Reading every EDA output in the final HTML

The HTML contains five visual EDA outputs and two printed diagnostic outputs. Each one answers a different question.

### Output A — Target-class bar chart

- **x-axis:** `C`, `CL`, and `D`;
- **y-axis:** number of patients;
- **exact counts:** `C=8,151`, `CL=317`, `D=3,532`;
- **visual message:** the `CL` bar is almost flat compared with the other two.

This confirms that the rare-class issue is not a small imbalance. `CL` is only 2.64% of the training set. The immediate workflow consequence is stratification: every validation fold must preserve the class proportions as closely as possible.

### Output B — Missing-values bar chart

The horizontal bars rank columns by missing percentage. The chart shows three distinct regimes:

1. `Triglyceride_Level` and `Cholesterol_Level` above 55%;
2. seven variables clustered around 43%;
3. `Platelet_Count` at 3.7% and `Prothrombin_Time` at 0.1%.

The seven bars clustered near 43% are the key visual clue. Independent missingness would not normally produce seven almost identical rates. This motivated the explicit joint-missingness count in the next output rather than an automatic “drop columns above 50%” rule.

### Output C — Numerical distributions by class

The density-normalized histograms show that:

- several laboratory variables are strongly right-skewed;
- many class distributions overlap substantially;
- some variables, especially liver-related measurements, shift toward more severe values for `D`;
- the `CL` curves are visibly noisy because only 317 observations contribute to them;
- `Clinical_Stage` appears as four discrete spikes rather than a continuous distribution.

The output supports two ideas at once. First, a linear model remains useful as a baseline but is unlikely to capture the complete relationship. Second, tree ensembles are appropriate because they can learn thresholds and interactions without requiring monotonic transformations or global scaling.

### Output D — Outcome composition within categorical levels

The 100% stacked bars use blue for `C`, yellow/orange for `CL`, and red for `D`. Several shifts are large enough to be visually meaningful:

- `Ascites_Indicator=Present`, `Liver_Enlargement=Present`, and `Spider_Angioma=Present` contain much larger `D` shares than the corresponding `Absent` levels;
- persistent edema has a very large `D` share, while `Edema_Status=None` has a much larger `C` share;
- `Clinical_Stage=4` has a much larger `D` proportion than stages 1–3;
- treatment-assignment groups have different outcome compositions;
- sex also shows a visible composition difference in this dataset.

These are associations in the competition data, not causal medical conclusions. The modeling consequence is simply that the categorical predictors contain signal and should be retained. The plot also reinforces why arbitrary integer label encoding would be unsafe for nominal variables.

### Output E — Train-versus-test distribution overlays

The first six numerical variables are overlaid for train and test. The distributions and printed summary statistics are broadly similar. For example:

| Feature | Train mean | Test mean | Train missing % | Test missing % |
|---|---:|---:|---:|---:|
| `Followup_Days` | 1978.274 | 1964.716 | 0.0 | 0.0 |
| `Patient_Age_Days` | 19311.575 | 19305.011 | 0.0 | 0.0 |
| `Bilirubin_Level` | 1.810 | 1.853 | 0.0 | 0.0 |
| `Cholesterol_Level` | 329.262 | 330.536 | 55.5 | 56.2 |
| `Albumin_Level` | 3.524 | 3.527 | 0.0 | 0.0 |
| `Copper_Level` | 74.769 | 74.914 | 43.8 | 44.5 |

This output does **not** prove the absence of distribution shift; it only says that no dramatic univariate shift is visible in the inspected variables. It supports using OOF validation as the primary criterion while keeping the leaderboard as a secondary check.

### Output F — Missingness by target table

The printed table establishes that missingness itself is predictive. The largest spreads are:

- `Cholesterol_Level`: 11.1 percentage points;
- `Triglyceride_Level`: 10.9 percentage points;
- `Copper_Level`: 6.7 percentage points.

This is why the Logistic Regression pipeline both imputes values and adds missing indicators, rather than replacing missing values and forgetting that they were absent.

### Output G — Joint missingness of the seven-column block

The exact count is almost all-or-nothing:

| Number missing from the seven fields | Rows | Share |
|---:|---:|---:|
| 0 | 6,737 | 56.1% |
| 1 | 79 | 0.7% |
| 2 | 3 | 0.0% |
| 3 | 1 | 0.0% |
| 4 | 1 | 0.0% |
| 5 | 2 | 0.0% |
| 6 | 30 | 0.2% |
| 7 | 5,147 | 42.9% |

The `0` and `7` groups alone cover 99.0% of the data. This confirms a structural recording pattern and justifies the `in_trial` completeness proxy. It does not independently confirm the clinical cause of that pattern.

## EDA decision summary

| EDA finding | Modeling action |
|---|---|
| `CL` is extremely rare | Shared stratified folds |
| Eleven features contain missing values | Retain and model missingness |
| Missingness differs by outcome | Missing indicators / explicit missing categories |
| Seven columns disappear together | Add `in_trial` proxy |
| Numerical relationships appear nonlinear | Test boosting models |
| Numerical scales differ | Scale Logistic Regression only |
| Categorical values are nominal | One-hot or native categorical treatment |
| Train/test appear broadly similar | Trust OOF primarily, LB secondarily |

---

# 10. Step 5 — Feature matrix construction

## Purpose

Step 5 converts the raw training and test DataFrames into aligned predictor matrices while avoiding learned transformations that could leak validation information.

## Objects created

```text
X      → training predictors
X_test → test predictors
y      → encoded training labels
```

The target and ID columns are removed from the predictors.

The ID is needed later to build the submission file, but it is not used as a predictive feature.

## Engineered feature: `in_trial`

The block-completeness proxy is created from the seven-column pattern found in EDA.

This is intended to give the models one clear structural signal rather than requiring them to rediscover the same condition repeatedly across several missing columns. Because no formal ablation was run, the document does not claim that this feature alone caused a measured score improvement.

## Explicit categorical missing level

Missing categorical values are replaced by:

```text
__MISSING__
```

This is different from mode imputation. Mode imputation would place a missing record into the most common real category, potentially creating false information.

An explicit missing category preserves the distinction between:

- an observed clinical category;
- no recorded value.

## Why median imputation and scaling are not performed here

Median values, means, standard deviations, and category vocabularies are statistics learned from data.

Learning them before cross-validation would allow validation rows to influence their own transformation.

For Logistic Regression, these operations are therefore placed inside a scikit-learn `Pipeline` and fit independently within each training fold.

## Final dimensions and exact notebook output

After adding `in_trial`:

```text
Feature matrix (train) : (12000, 19)
Feature matrix (test)  : (10000, 19)
Target vector          : (12000,)   [C=8151, CL=317, D=3532]

Numeric features       : 12
Categorical features   : 7

in_trial breakdown:
  in trial        6,737
  not in trial    5,263
```

The 19 predictors include:

- 12 numerical features, including `in_trial`;
- 7 categorical features.

The `5,263` rows coded as `in_trial=0` include both the 5,147 rows with all seven block fields missing and the small number of partial-pattern rows. In other words, the feature means **“all seven fields are present”**, not a verified enrollment label.

The matching train/test shapes confirm that feature construction created the same columns in the same order for both datasets. This is essential because a model fitted on one column layout cannot safely predict a matrix with a different layout.

## Safety checks

The notebook asserts that:

- training and test columns are identical and in the same order;
- encoded target values contain no missing entries.

### Step 5 outcome

The project now has aligned, model-ready feature matrices with structural missingness preserved and no pre-validation parameter fitting.

---

# 11. Step 6 — Validation strategy

## 11.1 Why cross-validation is necessary

A single train/validation split can be unusually easy or difficult. This risk is especially important when one class has only 317 samples.

Five-fold cross-validation divides the 12,000 training rows into five groups. The process is repeated five times:

```text
Round 1: folds 2–5 train, fold 1 validates
Round 2: folds 1,3,4,5 train, fold 2 validates
Round 3: folds 1,2,4,5 train, fold 3 validates
Round 4: folds 1,2,3,5 train, fold 4 validates
Round 5: folds 1–4 train, fold 5 validates
```

Every row is used:

- for training in four rounds;
- for validation in one round.

## 11.2 Why stratification is required

`StratifiedKFold` preserves the class ratio in every fold.

Each validation fold contains approximately:

```text
68% C
2.6–2.7% CL
29.4% D
```

Without stratification, one fold might contain too few `CL` rows, making the metric unstable and the per-class behavior difficult to compare.

## 11.3 What OOF predictions mean

OOF stands for **out-of-fold**.

For every training row, its OOF prediction comes from the one fold in which that row was withheld from training.

Example:

```text
Patient A belongs to validation fold 3.
The fold-3 model is trained without Patient A.
The fold-3 model predicts Patient A.
That probability vector becomes Patient A's OOF prediction.
```

After all five folds, every training row has one honest probability prediction.

The complete OOF matrix has shape:

```text
12,000 × 3
```

It supports:

- fair model comparison;
- blend optimization;
- class-level error analysis;
- calibration analysis.

## 11.4 Test predictions across folds

Each of the five fold models predicts all 10,000 test rows.

For a single model family:

```text
final test probability
= average of fold-1, fold-2, fold-3, fold-4, and fold-5 probabilities
```

This uses all five trained models and is usually more stable than relying on one fold model.

## 11.5 The exact fold output

The fold-construction cell prints:

| Fold | Training rows | Validation rows | C | CL | D |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,600 | 2,400 | 68.0% | 2.6% | 29.4% |
| 2 | 9,600 | 2,400 | 67.9% | 2.6% | 29.5% |
| 3 | 9,600 | 2,400 | 67.9% | 2.6% | 29.5% |
| 4 | 9,600 | 2,400 | 67.9% | 2.7% | 29.4% |
| 5 | 9,600 | 2,400 | 67.9% | 2.7% | 29.4% |

This output confirms that stratification worked. Each fold contains roughly 63–64 `CL` rows rather than leaving the minority class concentrated in only one split.

## 11.6 The `run_cv` driver

The notebook uses one shared function, `run_cv`, for every model.

The model-specific training function is responsible for:

- fold-specific preprocessing;
- fitting the algorithm;
- returning validation and test probabilities.

`run_cv` is responsible for:

- looping through folds;
- placing validation predictions in the correct original rows;
- averaging test predictions;
- calculating fold and overall OOF Log Loss;
- recording runtime and results in `RESULTS`.

This separation prevents model-specific code from changing the validation procedure.

A simplified version of the logic is:

```python
oof = np.zeros((len(X), 3))
test_pred = np.zeros((len(X_test), 3))

for tr_idx, va_idx in FOLDS:
    proba_va, proba_te = train_fn(
        X.iloc[tr_idx], y[tr_idx],
        X.iloc[va_idx], X_test, y[va_idx]
    )

    oof[va_idx] = proba_va
    test_pred += proba_te / N_FOLDS

cv_score = log_loss(y, oof, labels=[0, 1, 2])
```

## 11.7 Why validation predictions are placed, not averaged

Each validation fold contains a different set of rows. Therefore the five `proba_va` matrices cannot be averaged row by row.

Instead:

```python
oof[va_idx] = proba_va
```

returns every validation prediction to the row's original location.

After fold 1, only fold-1 rows contain predictions. After all five folds, every one of the 12,000 training rows contains exactly one OOF prediction.

## 11.8 Why test predictions are averaged

Every fold model predicts the **same 10,000 test rows**, so those matrices are aligned and can be averaged:

```python
test_pred += proba_te / 5
```

This produces one probability vector per test patient using information from all five fitted fold models.

## 11.9 Fold score versus overall OOF score

The notebook records both:

- Log Loss for each validation fold;
- Log Loss on the complete 12,000-row OOF matrix.

The complete OOF score is calculated directly:

```python
cv_score = log_loss(y, oof, labels=[0, 1, 2])
```

Because every fold contains 2,400 rows, this value is numerically equivalent to the equally weighted mean of the five fold losses, apart from rounding. Calculating it on the complete OOF matrix is still the more general and safer implementation.

## 11.10 What one OOF score represents

`oof` and `cv_score` are different objects:

```text
oof
= 12,000 × 3 probability matrix

cv_score
= one number summarizing the quality of that entire matrix
```

For each row, Log Loss uses only the probability assigned to the true class. A true `CL` row receiving `P(CL)=0.01` incurs a much larger penalty than one receiving `P(CL)=0.60`, even if both models ultimately choose the wrong top class.

This is why OOF probabilities—not only hard labels—are preserved throughout the project.

### Step 6 outcome

The project now has a common, leakage-controlled evaluation framework. Every model that follows is judged under the same conditions.

---

# 12. Step 7 — Logistic Regression baseline

## Purpose

Logistic Regression provides a simple, interpretable reference point.

A baseline answers:

> How well can a carefully prepared linear model perform before adding nonlinear complexity?

A complex model is valuable only if it produces a meaningful improvement over this baseline.

## Fold-specific preprocessing

Logistic Regression cannot directly handle missing values or text categories. Its preprocessing pipeline includes:

### Numerical branch

1. Median imputation
2. Missing-value indicators
3. Standard scaling

### Categorical branch

1. One-hot encoding
2. Ignore unseen categories safely
3. Group extremely rare levels when configured

## Why median imputation?

The median is more robust to extreme values than the mean. Since several laboratory variables are right-skewed, median imputation is a practical baseline choice.

The median is learned only from the training portion of each fold.

## Why missing indicators?

Imputation fills a numerical gap, but it removes the knowledge that the value was originally missing.

A missing indicator restores that information:

```text
value_was_missing = 1 or 0
```

This implements the EDA finding that missingness differs by outcome.

## Why scaling?

Logistic Regression estimates coefficients through numerical optimization. Predictors on very different scales can make optimization less stable and regularization less balanced.

Standard scaling transforms numerical variables to comparable scales.

## Why one-hot encoding?

Nominal categories have no valid numeric order. One-hot encoding creates a separate binary indicator for each level without implying that one treatment or condition is greater than another.

## Regularization

The baseline uses `C=0.5`. In scikit-learn, a smaller `C` means stronger regularization.

Regularization limits coefficient magnitude and helps control overfitting after one-hot encoding expands the feature space.

## Exact output from the final notebook

| Fold | Log Loss |
|---:|---:|
| 1 | 0.46607 |
| 2 | 0.45791 |
| 3 | 0.46225 |
| 4 | 0.43779 |
| 5 | 0.45580 |
| **Overall OOF** | **0.45596** |

```text
Fold SD = 0.00975
Runtime = approximately 0.4 seconds in the final recorded run
```

Fold 4 is substantially easier than fold 1 for this model. This does not invalidate cross-validation; it demonstrates why one split would be unreliable. The overall OOF score combines all 12,000 withheld-row predictions and is therefore the primary value.

## Interpretation

The baseline is stable and fast, but its score indicates that the relationship between predictors and outcomes is not adequately captured by a linear decision surface.

### Decision

Retain Logistic Regression as the reference model, but test nonlinear tree-based methods.

---

# 13. Step 8 — XGBoost

## Purpose

XGBoost tests the hypothesis that nonlinear thresholds and interactions among clinical variables are important.

## Plain-language intuition

XGBoost builds many small decision trees sequentially.

```text
Tree 1 makes an initial prediction.
Tree 2 focuses on mistakes left by Tree 1.
Tree 3 focuses on mistakes left by the previous trees.
...
The final probability is the combined contribution of all trees.
```

## Why it fits this dataset

- It models nonlinear relationships.
- It automatically captures interactions.
- It does not require numerical scaling.
- It can learn default directions for missing numerical values.
- It supports categorical variables in the configuration used by the notebook.

## Default configuration

The initial manually chosen configuration uses a small learning rate, moderate depth, row and feature subsampling, L2 regularization, and early stopping.

A high `n_estimators` value of 3,000 is only an upper limit.

## Early stopping

After each boosting round, the model checks validation Log Loss. If the metric does not improve for 150 rounds, training stops.

This allows the effective number of trees to adapt to each fold and reduces unnecessary overfitting.

## Exact output from the final notebook

| Fold | Log Loss |
|---:|---:|
| 1 | 0.37690 |
| 2 | 0.37036 |
| 3 | 0.37593 |
| 4 | 0.36208 |
| 5 | 0.38531 |
| **Overall OOF** | **0.37412** |

```text
Fold SD = 0.00768
Runtime = approximately 20.9 seconds in the final recorded run
```

The same broad difficulty pattern appears again: fold 4 is easiest and fold 5 is hardest. Because XGBoost uses the same rows as Logistic Regression in each fold, the improvement cannot be attributed to a luckier split.

## Improvement over baseline

```text
0.45596 − 0.37412 = 0.08184
```

This is approximately an 18% reduction relative to the Logistic Regression score.

## Interpretation

The large improvement supports the hypothesis that the dataset contains important nonlinear relationships and feature interactions.

### Decision

Keep XGBoost as a leading standalone model and blend candidate.

---

# 14. Step 9 — LightGBM

## Purpose

LightGBM provides a second strong boosting approach with a different tree-growth strategy.

## XGBoost versus LightGBM

A simplified distinction is:

- XGBoost typically grows trees level by level;
- LightGBM grows the leaf that offers the greatest immediate loss reduction.

This can create different tree structures and slightly different error patterns.

The value of LightGBM is therefore not only its standalone score. It may also contribute diversity to an ensemble.

## Complexity controls

Important settings include:

- `num_leaves` to control tree capacity;
- `min_child_samples` to prevent splits on very small groups;
- row and feature subsampling;
- L2 regularization;
- early stopping.

## Exact output from the final notebook

| Fold | Log Loss |
|---:|---:|
| 1 | 0.37721 |
| 2 | 0.37117 |
| 3 | 0.37736 |
| 4 | 0.36234 |
| 5 | 0.38514 |
| **Overall OOF** | **0.37464** |

```text
Fold SD = 0.00758
Runtime = approximately 38.4 seconds in the final recorded run
```

Its fold pattern is almost identical to XGBoost's. This suggests that fold difficulty is driven largely by the patient cases in each validation split, not by a unique failure of one library.

## Interpretation

LightGBM trails XGBoost by only 0.00052, much less than either model's fold-to-fold variability. They should be treated as near-ties rather than as evidence that one is universally superior.

### Decision

Keep LightGBM as a leading standalone model and blend candidate.

---

# 15. Step 10 — CatBoost

## Purpose

CatBoost is tested because the dataset contains several categorical predictors.

## Native categorical handling

CatBoost can accept categorical columns as strings through `Pool` objects. It is designed to reduce leakage risks associated with naive target encoding.

For low-cardinality variables, the configured `one_hot_max_size` may cause CatBoost to one-hot encode some categories internally.

## Why no external one-hot encoding is used

Externally one-hot encoding the data would prevent CatBoost from applying its own categorical mechanisms. The notebook therefore passes the original categorical columns directly.

## Overfitting control

CatBoost uses its own early-stopping mechanism, with an overfitting detector and `use_best_model=True`.

This restores the iteration that achieved the best validation score rather than automatically using the final iteration.

## Exact output from the final notebook

| Fold | Log Loss |
|---:|---:|
| 1 | 0.38726 |
| 2 | 0.37929 |
| 3 | 0.38791 |
| 4 | 0.37240 |
| 5 | 0.39050 |
| **Overall OOF** | **0.38347** |

```text
Fold SD = 0.00669
Runtime = approximately 8.0 seconds in the final recorded run
```

CatBoost is the most stable of the default boosting models by fold standard deviation, but its mean OOF Log Loss is worse. Low variability does not compensate for consistently weaker probability estimates.

## Interpretation

CatBoost strongly outperforms the linear baseline but underperforms XGBoost and LightGBM.

This is a useful result because theoretical suitability does not guarantee empirical superiority. The categorical variables have low cardinality, and the dataset's strongest signal may be driven more by numerical thresholds and interactions than by CatBoost's specialized categorical machinery.

CatBoost nevertheless has the lowest fold standard deviation among the default boosting models, indicating stable behavior across the shared folds.

### Decision

Keep CatBoost as an experimental and ensemble candidate, but allow validation evidence to determine its final influence.

---

# 16. Step 11 — Model comparison

## 16.1 How to read the model-comparison figure

The left panel is a horizontal bar chart of overall OOF Log Loss. Lower bars are better, and the error bars show the standard deviation across the five folds. The right panel plots the five individual fold losses for every model.

Three details matter:

1. **The nonlinear models form a separate performance tier.** XGBoost, LightGBM, and CatBoost all lie far below Logistic Regression.
2. **XGBoost and LightGBM are nearly tied.** Their mean difference is only `0.00052`, far smaller than either fold standard deviation (`0.00768` and `0.00758`).
3. **The folds move together.** Fold 4 is easiest for all four models, while fold 5 is among the hardest for all boosting models. This shared movement suggests that some validation partitions contain intrinsically harder patients rather than one library failing uniquely.

The chart therefore supports moving forward with both XGBoost and LightGBM. Declaring XGBoost categorically superior would overstate what the evidence shows.

## Default-model results

| Rank | Model | OOF Log Loss | Fold SD | Runtime | Improvement vs. baseline |
|---:|---|---:|---:|---:|---:|
| 1 | XGBoost | 0.37412 | 0.00768 | 20.9 s | 0.08184 (17.95%) |
| 2 | LightGBM | 0.37464 | 0.00758 | 38.4 s | 0.08132 (17.84%) |
| 3 | CatBoost | 0.38347 | 0.00669 | 8.0 s | 0.07249 (15.90%) |
| 4 | Logistic Regression | 0.45596 | 0.00975 | 0.4 s | Reference |

## Main findings

### Finding 1 — All boosting models beat the linear baseline

This confirms that nonlinear decision boundaries and feature interactions are important.

### Finding 2 — XGBoost and LightGBM are effectively tied

Their score difference is much smaller than the observed variation among folds.

### Finding 3 — CatBoost is useful but weaker

It should not receive equal influence simply because it is designed for categorical data.

### Finding 4 — Fold difficulty is shared

The models show similar patterns of easier and harder folds. Fold 4 is consistently easier and fold 5 is consistently harder for all three boosting models. This suggests that some folds contain intrinsically harder patient cases rather than one algorithm failing uniquely on a particular split.

### Finding 5 — Runtime is not the selection criterion

CatBoost is fastest among the default boosting models, but it is not selected because the competition metric is Log Loss. Runtime is documented for reproducibility and operational awareness, while predictive probability quality remains the main criterion.

### Decision

Proceed to:

- systematic tuning of all three boosting models;
- OOF probability blending;
- detailed class-level diagnostics.

---

# 17. Step 12 — Optuna hyperparameter tuning

Step 12 is the most computationally expensive and methodologically detailed stage.

## 17.1 Why tune hyperparameters?

The default boosting settings were selected manually. They produced strong results, but there was no guarantee that they were close to the best configuration for this dataset.

A **hyperparameter** is a setting chosen before training, such as:

- learning rate;
- tree depth;
- number of leaves;
- minimum child size;
- row subsampling;
- column subsampling;
- regularization strength.

The model learns tree splits and leaf values from data, but it does not decide these broader settings automatically.

## 17.2 Why use Optuna?

A grid search tests every combination on a fixed grid and can become expensive quickly.

A random search samples combinations without using the outcomes of previous trials.

Optuna's TPE sampler is adaptive. It uses previous trial results to guide later proposals toward promising regions of the search space.

This does not guarantee that later trials always improve or that the global optimum is found. It provides a more efficient search strategy under a limited computational budget.

## 17.3 Why use two rounds?

The chosen search range is itself a hypothesis. For example:

```text
num_leaves between 15 and 63
```

If the best scout result lands at 15, Optuna may be signaling that smaller values deserve exploration.

Rather than spend the full budget immediately, the project uses:

### Round 1 — Scout search

- 20 trials;
- 2 seeds: 42 and 123;
- purpose: test whether the initial search ranges appear too narrow.

### Round 2 — Final search

- 60 trials;
- 3 seeds: 42, 123, and 2024;
- purpose: search the corrected ranges with a more stable objective.

This design caught two real range issues:

- LightGBM `num_leaves` reached the lower scout boundary;
- CatBoost `depth` reached the lower scout boundary.

Both ranges were widened downward before the final round.

## 17.4 Repeated cross-validation objective

For each hyperparameter trial:

```text
for each seed
    create five stratified folds
    for each fold
        train model
        calculate validation Log Loss
average all fold scores across seeds
return the mean to Optuna
```

The final-round objective averages 15 fits per trial:

```text
3 seeds × 5 folds = 15 fits
```

These scores are repeated evaluations over overlapping observations, not 15 statistically independent samples. Their mean is used as a practical stability-oriented optimization target.

## 17.5 Early stopping within every trial

Each trial allows up to 3,000 boosting rounds, but early stopping determines the effective number of trees.

A parameter combination with a small learning rate may need more rounds. A more aggressive combination may stop earlier.

This avoids treating the number of trees as a fixed manual guess.

## 17.6 Boundary diagnostics

After each search, the notebook checks whether the best value is at or near a declared range edge.

- Integer parameters are checked for exact boundary matches.
- Floating-point parameters are inspected with a near-edge tolerance.

A boundary hit is a warning, not proof that the true optimum lies outside the range. It indicates that the range deserves review.

The final technical report honestly records a remaining limitation: the integer check is less sensitive to values trending toward an edge without landing exactly on it.

## Reading the search outputs correctly

The “best Log Loss” printed by an Optuna study is not the same object as the later standard OOF score.

- **Optuna study value:** average fold score across several seeds, used to rank hyperparameter candidates.
- **Refit OOF score:** one complete 12,000-row OOF matrix created on the standard seed-42 folds, used for direct comparison, blending, and diagnostics.

For example, XGBoost's final-round Optuna value is approximately `0.37140`, while its standard refit OOF score is `0.37149`. The small difference is expected because the two numbers summarize different evaluation procedures.

## Exact scout and final search outputs

| Model | Scout-round result | What the scout revealed | Final-round result | Interpretation |
|---|---|---|---|---|
| XGBoost | 0.37127 | `colsample_bytree=0.502`, very near the lower bound 0.5 | 0.37140 with `colsample_bytree=0.516` | Soft edge concern did not persist |
| LightGBM | 0.37023 | `num_leaves=15`, exactly the lower bound | 0.37089 with `num_leaves=8` | The original range was genuinely too narrow |
| CatBoost | 0.37970 | `depth=4`, exactly the lower bound | 0.37901 with `depth=3` | Shallower trees fit this dataset better; depth still trends low |

The LightGBM result is the clearest validation of the two-round design: the final optimum (`num_leaves=8`) lies outside the original search range. A single search using the initial range could never have found it.

## 17.7 Final tuned hyperparameters

### XGBoost

| Parameter | Tuned value |
|---|---:|
| Learning rate | 0.0238 |
| Maximum depth | 4 |
| Minimum child weight | 14 |
| Row subsample | 0.739 |
| Column subsample | 0.516 |
| L2 regularization | 0.981 |

### LightGBM

| Parameter | Tuned value |
|---|---:|
| Learning rate | 0.0327 |
| Number of leaves | 8 |
| Minimum child samples | 30 |
| Feature fraction | 0.547 |
| Bagging fraction | 0.707 |
| L2 regularization | 1.145 |

### CatBoost

| Parameter | Tuned value |
|---|---:|
| Learning rate | 0.0580 |
| Depth | 3 |
| L2 leaf regularization | 1.81 |
| Subsample | 0.684 |

The tuned solutions favor relatively small trees, strong minimum-child constraints, and substantial feature or row subsampling. This suggests that controlling variance was more valuable than increasing model complexity.

## 17.8 Why refitting is required after Optuna

The Optuna objective returns one average score for a trial. It does not preserve the complete OOF and test probability matrices needed by later steps.

After selecting the best hyperparameters, each model is therefore refit on the notebook's standard seed-42 folds.

This serves two purposes:

1. create OOF and test predictions for blending and error analysis;
2. compare tuned and default models on exactly the same fold assignment.

## 17.9 Tuning results

| Model | Default OOF | Tuned OOF | Improvement |
|---|---:|---:|---:|
| XGBoost | 0.37412 | 0.37149 | 0.00263 |
| LightGBM | 0.37464 | 0.37094 | 0.00370 |
| CatBoost | 0.38347 | 0.37952 | 0.00395 |

All three models improve after tuning.

LightGBM becomes the strongest tuned standalone model.

### Exact tuned fold outputs

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF | Fold SD | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost tuned | 0.37289 | 0.36834 | 0.37178 | 0.36164 | 0.38282 | 0.37149 | 0.00689 | 30 s |
| LightGBM tuned | 0.37396 | 0.36840 | 0.37250 | 0.35897 | 0.38088 | 0.37094 | 0.00721 | 28 s |
| CatBoost tuned | 0.37959 | 0.37721 | 0.38604 | 0.36864 | 0.38612 | 0.37952 | 0.00648 | 7 s |

The ranking is consistent across both the repeated-seed Optuna objective and the standard refit: LightGBM is strongest, XGBoost is very close, and CatBoost remains weaker.

The tuned configurations are generally smaller and more regularized than the initial defaults. This supports the idea that the dataset benefits from controlled model capacity rather than deeper trees.

### Runtime breakdown for tuning

The final HTML records:

| Study | Runtime |
|---|---:|
| XGBoost scout | 21.3 min |
| XGBoost final | 98.0 min |
| LightGBM scout | 43.1 min |
| LightGBM final | 136.4 min |
| CatBoost scout | 6.2 min |
| CatBoost final | 27.5 min |
| Baselines, tuned refits, and blends | 2.2 min |
| **Recorded total** | **334.8 min (5.58 h)** |

The seed-averaging runtime is not included in this total because the temporary seed-specific entries were removed from `RESULTS` after averaging, so their runtimes were not retained in the final summary.

## 17.10 Why the blend weights must be recalculated

The original blend weights were optimized for default models. Tuning changes each model's probability surface and error pattern.

Reusing old weights would assume that the relative strengths and complementarities remained unchanged.

The project therefore re-optimizes the weights using the tuned OOF matrices.

## 17.11 Nelder–Mead blend optimization

The blend objective takes candidate weights, normalizes them to a nonnegative sum of one, combines the three OOF probability matrices, and returns multiclass Log Loss.

Nelder–Mead is used because:

- only three weights are being optimized;
- no gradient is required;
- the computation is fast relative to model training.

Limitations:

- Nelder–Mead does not guarantee a global optimum;
- taking absolute values and normalizing creates a practical but non-smooth parameterization;
- weights are optimized and evaluated on the same OOF predictions, making a small gain potentially optimistic.

## 17.12 Tuned blend result

| Component | Weight |
|---|---:|
| XGBoost | 0.4308 |
| LightGBM | 0.5683 |
| CatBoost | 0.0008 |

```text
Tuned blend OOF Log Loss = 0.37010
Public Leaderboard       = 0.38042
```

The optimizer assigns CatBoost almost zero weight. The final ensemble therefore behaves approximately as a two-model XGBoost–LightGBM blend.

This is not a defect. It is evidence that the weaker model adds almost no incremental value after tuning.

The blend improves on tuned LightGBM by:

```text
0.37094 − 0.37010 = 0.00084
```

This is a real measured OOF gain, but it is small. Because the weights were selected and scored on the same OOF data, the gain should be interpreted as promising rather than as an unbiased estimate of future improvement. The Public Leaderboard moving from the default blend's 0.38364 to 0.38042 provides secondary support that the tuned progression was genuinely useful.

## 17.13 Registration in `RESULTS`

The tuned blend is saved using the same keys as every other model:

```text
oof
cv_score
test_pred
```

This enables the generic downstream code to analyze and export it automatically.

### Step 12 decision

Keep the tuned blend as the final reliable submission candidate.

---

# 18. Step 13 — Seed averaging

## 18.1 Purpose

The tuned models in Step 12 are refit and evaluated on the standard seed-42 folds. Step 13 asks:

> Are the predictions stable if the data split and model randomness change?

## 18.2 What changes when the seed changes?

The notebook's global seed influences:

1. how `StratifiedKFold` assigns rows to folds;
2. the internal random state of the boosting algorithms.

Therefore, Step 13 is a combined **split-and-training-seed stability experiment**. It is not purely a test of fold assignment.

## 18.3 What remains fixed?

The tuned hyperparameters from Step 12 remain unchanged.

This is important because the question is not, “Can a new search find different settings?” The question is, “How stable are these selected settings across randomness?”

## 18.4 Procedure

The seed-42 predictions from Step 12 are reused. The models are then retrained under seeds 123 and 2024.

For each model:

```text
OOF average = mean(seed 42 OOF, seed 123 OOF, seed 2024 OOF)
Test average = mean(seed 42 test, seed 123 test, seed 2024 test)
```

The three seed-averaged model outputs are then blended with newly optimized weights.

## 18.5 Why global state is backed up

`run_cv` and the tuned training functions read global `SEED` and `FOLDS` values.

Before the experiment, the notebook stores the originals. After training the additional seeds, it restores them immediately.

Without restoration, later cells could silently run under an unintended fold configuration.

## 18.6 Exact per-seed and averaged results

| Model | Seed 42 OOF | Seed 123 OOF | Seed 2024 OOF | Three-seed averaged OOF |
|---|---:|---:|---:|---:|
| XGBoost | 0.37149 | 0.37119 | 0.37152 | **0.36830** |
| LightGBM | 0.37094 | 0.37032 | 0.37141 | **0.36734** |
| CatBoost | 0.37952 | 0.37753 | 0.37998 | **0.37587** |

The individual seed scores are similar, which suggests that the selected hyperparameters are not dependent on one uniquely lucky split.

The fold standard deviations for the new seeds are larger—roughly 0.011 to 0.014—than for seed 42. That reveals meaningful split sensitivity even when the overall seed-level OOF scores remain similar.

### Why the averaged score can beat every individual seed

For each training row, the notebook has three honest OOF probability vectors, one from each seed's fold system. It averages those vectors before calculating Log Loss.

Probability averaging can reduce variance and soften confident mistakes. Because Log Loss is nonlinear, the loss of the average probability is not required to equal the average of the three losses. A smoother averaged prediction can therefore score better than every individual run.

### Why averaging OOF predictions from different folds is valid here

Each row is out-of-fold once under seed 42, once under seed 123, and once under seed 2024. None of the three predictions for that row comes from a model that trained on the row under the corresponding fold system.

Therefore, averaging the three row-aligned OOF predictions preserves the out-of-fold principle.

## 18.7 Blend result

```text
Single-seed tuned blend OOF = 0.37010
Seed-averaged blend OOF     = 0.36713
```

At first glance, seed averaging appears superior.

The code cell prints:

```text
Seed-averaging IMPROVED the blend. Consider using it.
```

That message is intentionally understood as an **OOF-only provisional conclusion**. The code at that point has not yet incorporated the later Public Leaderboard evidence.

However, the Public Leaderboard comparison is:

```text
Single-seed tuned blend Public LB = 0.38042
Seed-averaged blend Public LB     = 0.38067
```

The seed-averaged submission is slightly worse on the available hidden-label sample.

## 18.8 How the disagreement is interpreted

The two signals point in opposite directions:

- OOF favors seed averaging;
- the Public Leaderboard slightly favors the simpler tuned blend.

The differences are small. Comparing the aggregate OOF gap directly with fold standard deviation is only a heuristic, not a formal statistical test.

A more rigorous future analysis could use:

- paired per-row Log Loss differences;
- bootstrap confidence intervals;
- repeated-CV distributions;
- additional seed sets.

## 18.9 Final decision

Seed averaging is treated as **inconclusive**, not as a proven failure.

The simpler tuned blend is retained because:

- its OOF and Public LB improvements over the default blend agree;
- the seed-averaged gain is small and externally inconsistent;
- seed averaging adds training and operational complexity;
- the available evidence is insufficient to claim a robust improvement.

The Public Leaderboard is used as secondary evidence, not as the only model-selection rule.

A cleaner future version of the notebook should replace the provisional print message with wording such as:

```text
Seed averaging improved OOF. Final selection remains pending external
validation and uncertainty analysis.
```

This would keep the code output consistent with the final documented decision.

### Step 13 outcome

The experiment identifies a real uncertainty in model selection and demonstrates why small validation gains should be interpreted cautiously.

---

# 19. Step 14 — Default-model blending

Step 14 preserves the original ensemble experiment using the default boosting models. It remains valuable as a baseline for understanding why the tuned blend was designed the way it was.

## 19.1 Why blend models?

If several models make different mistakes, averaging their probability estimates can reduce model-specific error.

A blend is most useful when its members are:

- individually strong;
- not perfectly correlated;
- differently biased.

Logistic Regression is excluded because it is materially weaker and would likely dilute the stronger probability estimates.

## 19.2 Correlation analysis

The notebook compares the models' OOF probabilities for `Status_D`.

Pairwise correlations range from approximately 0.9911 to 0.9969.

These values are very high, meaning the models generally agree. The original notebook text used 0.98 as a rough diversity reference, but the measured correlations are above that threshold. Therefore, any blend gain should be expected to be modest rather than large.

This is an important correction in interpretation:

> High correlation does not make blending impossible, but it limits the amount of independent error available to cancel.

## 19.3 Equal-weight blend

The first blend gives each model one-third weight:

```text
(XGBoost + LightGBM + CatBoost) / 3
```

Result:

```text
Equal blend OOF = 0.37431
Best member OOF = 0.37412 (XGBoost)
```

The equal blend is slightly worse than XGBoost alone.

## 19.4 Why equal weighting failed

CatBoost is weaker than XGBoost and LightGBM. Giving it one-third of the blend gives it too much influence.

This demonstrates:

> Ensembling does not automatically improve performance. Member strength and weight allocation matter.

## 19.5 Optimized default blend

Nelder–Mead finds the following approximate weights:

| Model | Weight |
|---|---:|
| XGBoost | 0.5316 |
| LightGBM | 0.4289 |
| CatBoost | 0.0396 |

Result:

```text
Optimized default blend OOF = 0.37330
Public Leaderboard          = 0.38364
```

The optimizer reduces CatBoost's influence from 33.3% to approximately 4%.

### Step 14 decision

The optimized default blend is kept as an important experiment but is later superseded by the Optuna-tuned blend.

---

# 20. Step 15 — Error analysis

A strong overall metric can hide serious class-specific weaknesses. Step 15 investigates where the model fails.

## 20.1 Why compare three stages?

The updated error analysis compares:

1. **Before tuning:** default XGBoost, the strongest default standalone model;
2. **After tuning:** tuned LightGBM, the strongest tuned standalone model;
3. **Final blend:** the selected tuned blend.

This design asks whether tuning and blending improved the actual weakness of the model, not only the headline score.

## 20.2 Confusion matrices: exact counts and meaning

A confusion matrix converts each probability vector into one hard prediction by taking the largest probability. This is not the competition metric, but it reveals which classes cross the decision boundary.

### Default XGBoost

| True \ Predicted | C | CL | D | Recall |
|---|---:|---:|---:|---:|
| C | 7,627 | 28 | 496 | 0.936 |
| CL | 183 | 50 | 84 | 0.158 |
| D | 934 | 15 | 2,583 | 0.731 |

Of the 317 true transplant cases, only 50 are predicted as `CL`; 183 are pushed toward `C` and 84 toward `D`. The model therefore treats `CL` as an intermediate, overlapping state rather than a well-separated cluster.

### Tuned LightGBM

| True \ Predicted | C | CL | D | Recall |
|---|---:|---:|---:|---:|
| C | 7,617 | 27 | 507 | 0.934 |
| CL | 177 | 62 | 78 | 0.196 |
| D | 919 | 16 | 2,597 | 0.735 |

Tuning increases correct `CL` predictions from 50 to 62 and slightly improves `D` recall. The trade-off is ten fewer correctly hard-classified `C` rows. Since Log Loss evaluates probabilities rather than only argmax labels, this trade-off must be interpreted together with the probability metrics below.

### Final tuned blend

| True \ Predicted | C | CL | D | Recall |
|---|---:|---:|---:|---:|
| C | 7,621 | 26 | 504 | 0.935 |
| CL | 177 | 61 | 79 | 0.192 |
| D | 922 | 15 | 2,595 | 0.735 |

The blend sits between tuned LightGBM and default XGBoost in hard-label behavior. It correctly identifies one fewer `CL` case than tuned LightGBM, yet it achieves better `CL` mean Log Loss. That apparent contradiction is possible because the blend can assign more probability to the correct class across many rows without changing which class has the largest probability.

The row-normalized heatmaps make the imbalance visible fairly: each true-class row sums to one, so the small `CL` class is not visually overwhelmed by the thousands of `C` observations.

## 20.3 Class-specific metrics

| Class | Samples | Recall: before / tuned / blend | Mean P(true): before / tuned / blend | Mean Log Loss: before / tuned / blend |
|---|---:|---:|---:|---:|
| C | 8,151 | 0.936 / 0.934 / 0.935 | 0.861 / 0.861 / 0.861 | 0.199 / 0.198 / 0.198 |
| CL | 317 | 0.158 / 0.196 / 0.192 | 0.196 / 0.210 / 0.209 | 2.496 / 2.486 / 2.441 |
| D | 3,532 | 0.731 / 0.735 / 0.735 | 0.695 / 0.695 / 0.695 | 0.588 / 0.581 / 0.581 |

## 20.4 Interpretation by class

### Class C

`C` is predicted very well. It is the largest class and has the lowest per-row loss.

### Class D

`D` is handled reasonably well. Tuning produces a small improvement.

### Class CL

`CL` remains the main weakness:

- only 317 training samples;
- substantial feature overlap with `C` and `D`;
- recall improves from 15.8% to about 19%, but remains low;
- per-row Log Loss remains more than four times the `D` loss.

## 20.5 What tuning did and did not fix

Tuning improves aggregate Log Loss and slightly improves `CL` recall. It does not fundamentally solve the rare-class problem.

The final blend's `CL` recall (`0.192`) is slightly lower than tuned LightGBM's (`0.196`), yet its `CL` mean Log Loss is better (`2.441` versus `2.486`). This is an important metric lesson:

> A model can assign better probabilities to the true class without changing—or even while slightly reducing—the number of hard argmax predictions that are correct.

The competition rewards the probability improvement, not recall alone.

This suggests that the limiting factor is not merely hyperparameter choice. The current features may not contain enough information to separate transplant cases reliably.

## 20.6 Confidence analysis

The notebook also compares confidence for correct and incorrect predictions.

The exact summary is:

| Stage | Mean confidence when correct | Mean confidence when incorrect | Gap |
|---|---:|---:|---:|
| Default XGBoost | 0.890 | 0.718 | 0.172 |
| Tuned LightGBM | 0.890 | 0.717 | 0.173 |
| Final tuned blend | 0.889 | 0.716 | 0.173 |

The positive gap is useful: the model is generally less confident when it is wrong.

The near-identical values across all three stages show that tuning and blending did not materially change the overall confidence separation. Their gains came mainly from improved probability allocation on specific cases, especially some `CL` rows, rather than from a global change in confidence behavior.

However, these two averages alone do not prove calibration. The notebook also plots reliability curves by confidence bins. A complete calibration claim would require sufficient observations per bin, class-specific analysis, and an independent calibration-validation procedure.

### What the reliability curves add beyond the two mean-confidence numbers

The three reliability plots are visually similar. Their points generally follow the diagonal from moderate to high confidence, but the lowest-confidence bins lie below the perfect-calibration line. In those bins, observed accuracy is lower than stated confidence, indicating some overconfidence on ambiguous cases. Around the highest-confidence bins, the curves are much closer to the diagonal.

The confidence histograms show a strong green concentration near 0.9–1.0 for correct predictions, while incorrect predictions are spread broadly from roughly 0.45 through the high-confidence region. The overlap at high confidence explains why Log Loss remains sensitive: some errors are still made with substantial confidence.

The chart supports a restrained conclusion:

- the models separate easy and hard cases reasonably well;
- no dramatic global calibration failure is visible;
- the notebook does not perform an independent post-hoc calibration experiment, so the curves are diagnostics rather than proof of deployment-grade calibration.

## 20.7 Numerical stability with `np.clip`

Class-specific Log Loss uses the negative logarithm of the probability assigned to the true class.

If a probability is exactly zero, `log(0)` is undefined. The notebook clips probabilities to a very small positive lower bound before taking logarithms.

This does not meaningfully change normal predictions. It prevents numerical failure at extreme values.

### Step 15 conclusion

The final model's remaining error is concentrated in `CL`. Future improvement should focus on rare-class information, objective design, and uncertainty rather than continuing to tune the same feature set indefinitely.

---

# 21. Step 16 — Submission generation

## Purpose

Step 16 converts each model's 10,000 × 3 test probability matrix into a Kaggle-compatible CSV file.

## Required structure

```text
id
Status_C
Status_CL
Status_D
```

Each row must contain the test ID and a valid probability distribution.

## Automated checks

Before saving, `make_submission` verifies:

1. the number of rows equals the test-set length;
2. the columns exist in the correct order;
3. no value is missing;
4. the three probabilities sum approximately to one.

A small tolerance is used for the probability sum because floating-point arithmetic may produce values such as 0.9999999998 rather than exactly 1.0.

## Why automated checks matter

A model can be statistically strong and still produce an invalid submission because of:

- misordered class columns;
- a missing ID;
- an extra pandas index column;
- NaN values;
- non-normalized probabilities.

These checks prevent formatting mistakes from consuming submission attempts or producing misleading leaderboard scores.

## Exact submission-generation output

The generic loop writes one CSV for every registered model or blend:

```text
submission_logistic_regression.csv
submission_xgboost.csv
submission_lightgbm.csv
submission_catboost.csv
submission_xgboost_tuned.csv
submission_lightgbm_tuned.csv
submission_catboost_tuned.csv
submission_blend_tuned.csv
submission_blend_seed-averaged.csv
submission_blend_equal.csv
submission_blend_optimised.csv
```

The queue is sorted by OOF Log Loss, so the seed-averaged blend appears first. That ordering means only “best local OOF score”; it does not override the final experiment decision made after comparing Public Leaderboard evidence.

## Final selected file

```text
submission_blend_optimised_v3.csv
```

This file is generated in Step 16 from the `blend_test_tuned` matrix created in Step 12.

The first five rows shown in the final HTML are:

| id | Status_C | Status_CL | Status_D |
|---:|---:|---:|---:|
| 15000 | 0.892821 | 0.012067 | 0.095112 |
| 15001 | 0.902827 | 0.008595 | 0.088578 |
| 15002 | 0.982071 | 0.005109 | 0.012820 |
| 15003 | 0.905226 | 0.056038 | 0.038735 |
| 15004 | 0.305815 | 0.006424 | 0.687761 |

For ID 15004, for example, the highest probability is assigned to `D`, but the model still preserves uncertainty by assigning approximately 30.6% to `C`.

The seed-averaged candidate is also generated for experimental comparison but is not the primary final solution.

## File-naming duplication found during review

The final HTML produces a few duplicate or equivalent files:

- `submission_blend_tuned.csv` and `submission_blend_optimised_v3.csv` come from the same tuned-blend probability matrix;
- `submission_blend_seed-averaged.csv` and `submission_blend_seed_averaged.csv` are naming variants of the same seed-averaged candidate.

These duplicates do not change any score, but they make the output folder harder to understand. A cleaned repository should keep one canonical name for each candidate and clearly identify:

```text
submission_blend_optimised_v3.csv  ← actual primary submission
submission_blend_seed_averaged.csv ← rejected/inconclusive comparison candidate
```

Other experimental submissions can be archived in a separate `artifacts/submissions/` folder.

## Important code/documentation consistency correction

The reproducibility text printed later in Step 17 says that the last line of Step 12 calls `make_submission(...)`. In the final HTML, Step 12 actually:

1. creates `blend_test_tuned`;
2. registers `Blend (tuned)` in `RESULTS`.

The explicit call that writes `submission_blend_optimised_v3.csv` occurs in Step 16. This document uses the actual code execution order. The Step 17 print statement should be corrected to match it.

---

# 22. Step 17 — Experiment evidence and reproducibility

Step 17 saves the evidence needed to verify, explain, and extend the project.

## 22.1 Experiment table

`experiment_table.csv` records:

- experiment number;
- hypothesis;
- change made;
- OOF Log Loss;
- Public Leaderboard score when available;
- keep/reject decision.

This table makes the experimental progression auditable.

## 22.2 Per-class analysis

`per_class_analysis.csv` records class-level sample counts, recall, probability assigned to the true class, and Log Loss.

This prevents the rare-class weakness from being hidden behind one aggregate score.

## 22.3 OOF arrays

Files such as `oof_xgboost.npy` preserve raw OOF probability matrices.

They allow later work such as:

- alternative blending;
- calibration;
- bootstrap comparison;
- threshold analysis;
- error slicing;

without retraining the models.

## 22.4 Reproducibility summary

`report_summary.json` records items such as:

- Python and library versions;
- random seeds;
- data shapes;
- fold strategy;
- model scores;
- blend weights;
- runtimes;
- final submission identity.

The final HTML records this execution environment:

```text
Python          3.13.7
NumPy           2.3.3
pandas          2.3.3
scikit-learn    1.7.2
SciPy           1.16.2
XGBoost         3.3.0
LightGBM        4.7.0
CatBoost        1.2.10
Platform        macOS 15.6.1, arm64
Active run      local execution in VS Code + Jupyter
```

This distinction matters: the final HTML export and its timing figures come from the local MacBook run. The notebook also supports Google Colab/Drive, and portions of the project were tested in Kaggle-style CPU environments, but runtime comparisons should use the environment printed by the actual final run.

## 22.5 Final consistency check

The notebook verifies that all expected entries are present in `RESULTS` before printing the final table.

This protects against a common notebook problem: restarting the kernel and running only some cells, leaving stale or incomplete state in memory.

The final self-check prints:

```text
OK: all 11 expected models/blends are in RESULTS.
OK: 11 .npy files saved, matching 11 RESULTS entries.
OK: report_summary['results'] has 11 entries, matching RESULTS.
```

These messages confirm that:

1. no candidate expected by the final comparison is missing;
2. every registered model or blend has a corresponding saved OOF array;
3. the JSON summary and in-memory experiment registry describe the same set of results.

The temporary seed-123 and seed-2024 model entries are deliberately removed after averaging. Their predictions have already been incorporated into the seed-averaged arrays, and removing them prevents the final ranking from being cluttered by intermediate training runs.

## 22.6 Final ranking output

The notebook's final OOF-only ranking is:

| Rank | Candidate | OOF Log Loss |
|---:|---|---:|
| 1 | Blend (seed-averaged) | 0.36713 |
| 2 | Blend (tuned) | 0.37010 |
| 3 | LightGBM tuned | 0.37094 |
| 4 | XGBoost tuned | 0.37149 |
| 5 | Default optimized blend | 0.37330 |
| 6 | XGBoost default | 0.37412 |
| 7 | Default equal blend | 0.37431 |
| 8 | LightGBM default | 0.37464 |
| 9 | CatBoost tuned | 0.37952 |
| 10 | CatBoost default | 0.38347 |
| 11 | Logistic Regression | 0.45596 |

The output immediately clarifies that the first-ranked OOF candidate is not the actual primary submission. The selected final file remains Experiment 10 because the seed-averaged candidate's Public Leaderboard evidence disagreed with its OOF improvement.

## 22.7 Artifact inventory

The final run writes:

- `experiment_table.csv`;
- `per_class_analysis.csv`;
- `report_summary.json`;
- 11 OOF `.npy` arrays;
- standalone and blend submission CSVs.

The OOF arrays are approximately 281 KB each, while the submission CSVs are approximately 640–646 KB each. File size is not a quality metric; it simply confirms that the expected full-row arrays were serialized rather than empty placeholders.

### Step 17 outcome

The project ends with not only a final model, but also the artifacts needed to reproduce and audit how that model was selected.

---

# 23. Final model and experimental results

## 23.1 Experiment progression

| Exp | Experiment | OOF Log Loss | Public LB | Decision |
|---:|---|---:|---:|---|
| 0 | Logistic Regression baseline | 0.45596 | — | Baseline |
| 1 | XGBoost, default | 0.37412 | 0.38372 | Kept |
| 2 | LightGBM, default | 0.37464 | 0.38480 | Kept |
| 3 | CatBoost, default | 0.38347 | 0.38953 | Kept for comparison/blend |
| 4 | Equal default blend | 0.37431 | 0.38407 | Rejected |
| 5 | Optimized default blend | 0.37330 | 0.38364 | Superseded |
| 6 | Single-pass Optuna draft | Not preserved | 0.38161 | Superseded; not reproducible from final notebook |
| 7 | Two-round tuned XGBoost | 0.37149 | — | Kept |
| 8 | Two-round tuned LightGBM | 0.37094 | — | Kept |
| 9 | Two-round tuned CatBoost | 0.37952 | — | Kept |
| 10 | Optimized tuned blend | **0.37010** | **0.38042** | **Final reliable submission** |
| 11 | Seed-averaged tuned blend | 0.36713 | 0.38067 | Inconclusive; not selected |

### Experiment-log wording correction

The `experiment_table.csv` code in the final HTML labels Experiment 5 as “Selected — final submission.” That was true before Optuna tuning, but it is no longer the final project state. For consistency, its decision should read:

```text
Selected at the default-model stage; later superseded by Experiment 10.
```

Experiment 10 is the only candidate that should be labeled the final primary submission.

## 23.2 Why Experiment 10 is final

Experiment 10 is selected because:

- every component improves after systematic tuning;
- the tuned blend improves over the default blend on OOF;
- the public leaderboard moves in the same direction;
- the pipeline is simpler than the seed-averaged alternative;
- the result is fully represented and reproducible in the final notebook.

## 23.3 Final configuration

```text
Final prediction
= 0.4308 × XGBoost_tuned
+ 0.5683 × LightGBM_tuned
+ 0.0008 × CatBoost_tuned
```

The blend is effectively an XGBoost–LightGBM ensemble.

## 23.4 Final performance

```text
OOF Log Loss        = 0.37010
Public Leaderboard = 0.38042
```

The approximately 0.010 difference between OOF and Public LB indicates that the local validation is somewhat more optimistic than the public test subset, but the relative model ordering is reasonably consistent for the default-to-tuned progression.

No private-leaderboard result is claimed in this document because it was not provided in the project materials used to construct this documentation.

## 23.5 Evidence chain from idea to final decision

| Question or hypothesis | Experiment | Output | What it confirmed | Decision |
|---|---|---:|---|---|
| Is a linear model sufficient? | Logistic Regression | 0.45596 | Linear structure is inadequate | Move to boosting |
| Do nonlinear trees help? | XGBoost | 0.37412 | Large ~18% loss reduction | Keep tree models |
| Are multiple boosting libraries competitive? | XGBoost/LGBM/CatBoost | 0.37412 / 0.37464 / 0.38347 | XGB and LGBM are near-ties; CatBoost weaker | Blend selectively |
| Does equal averaging help? | Equal default blend | 0.37431 | No; weaker CatBoost weight drags the blend | Reject equal weighting |
| Do optimized default weights help? | Optimized default blend | 0.37330 | Small improvement; CatBoost falls to 4% | Keep as interim best |
| Were tuning ranges adequate? | Optuna scout rounds | boundary warnings | LGBM and CatBoost needed shallower ranges | Widen ranges |
| Does systematic tuning help? | Tuned models | 0.37149 / 0.37094 / 0.37952 | Every boosting model improves | Refit and reblend |
| Does tuned blending help? | Experiment 10 | 0.37010 / 0.38042 LB | OOF and LB improve together | Select final reliable model |
| Does seed averaging add stability? | Experiment 11 | 0.36713 / 0.38067 LB | OOF improves but LB worsens | Treat as inconclusive |
| Where does the final model fail? | Per-class analysis | `CL` recall ≈19%, loss 2.441 | Rare transplant class remains unresolved | Prioritize CL-specific future work |

---

# 24. What worked, what failed, and what was learned

## 24.1 What worked

### Careful data loading

Preserving `Edema_Status="None"` prevented a valid category from being converted into missing data.

### Shared stratified folds

The same folds made model comparisons fair and OOF blending valid.

### Missingness-aware modeling

The project retained high-missing features, preserved missing indicators, and modeled the seven-column structural pattern.

### Gradient boosting

Moving from Logistic Regression to XGBoost reduced OOF Log Loss by approximately 18%.

### Two-round tuning

The scout/final design identified narrow search ranges before spending the full budget.

### Honest diagnostics

The error analysis revealed that the aggregate score hides very poor performance on `CL`.

## 24.2 What failed or remained inconclusive

### Arbitrary ratio feature

A manually guessed ratio feature did not improve validation and was removed.

Lesson: feature engineering should be motivated by data structure and retained only when measured evidence supports it.

### Equal-weight blend

Equal averaging was worse than XGBoost alone because CatBoost received too much weight.

Lesson: ensembling weak and strong members equally can reduce performance.

### Seed averaging

Seed averaging improved OOF but slightly worsened the Public Leaderboard.

Lesson: a small validation gain is not automatically a reliable generalization gain.

### Hyperparameter tuning did not solve `CL`

Tuning improved the global metric but only modestly improved rare-class recall.

Lesson: a feature and data limitation cannot always be solved by searching model settings.

## 24.3 Most important learning

The project began with the expectation that choosing the strongest algorithm would be the main determinant of performance.

The results showed a more complete picture:

```text
Correct data interpretation
+ leakage-safe preprocessing
+ honest validation
+ appropriate nonlinear models
+ systematic tuning
+ cautious experiment interpretation
= reliable progress
```

Model choice produced the largest first improvement, but later reliability depended on validation design and disciplined interpretation.

## 24.4 My reasoning, in first-person form

I did not begin by choosing the most complicated model. I first asked whether the data were being read correctly, whether the target was imbalanced, whether missingness was random, and whether train and test looked broadly compatible.

The outputs then changed the pipeline step by step:

1. The target count showed only 317 `CL` rows, so I rejected ordinary random validation and used stratified folds.
2. The missingness tables showed that absence itself differed by outcome, so I did not simply drop or impute away that information.
3. The seven-column completeness pattern showed that missingness had structure, so I created one explicit proxy feature instead of asking every model to rediscover the same pattern.
4. The Logistic Regression score established that careful preprocessing alone was not enough.
5. The large boosting improvement confirmed my hypothesis that nonlinear thresholds and interactions mattered.
6. The near-tie between XGBoost and LightGBM suggested that both were useful, while CatBoost's weaker score warned me not to give every model equal influence.
7. The failed equal blend confirmed that an ensemble is not automatically better.
8. The Optuna boundary outputs showed that my original search ranges were part of the modeling hypothesis and could be wrong.
9. The tuned scores confirmed that smaller, more regularized trees generalized better on this dataset.
10. The seed-averaging disagreement taught me that a lower local score can still be too small or unstable to justify changing the final submission.
11. The per-class analysis confirmed that the remaining problem was concentrated in `CL`, so further generic tuning was unlikely to be the most efficient next step.

This is the central workflow I wanted the project to demonstrate:

```text
I formed a hypothesis
→ wrote code to test it
→ read the output
→ interpreted what the output could and could not prove
→ changed the next modeling decision
→ kept failed experiments as evidence
```

The final score is therefore not presented as an isolated leaderboard number. It is the endpoint of a traceable sequence of decisions.

---

# 25. Limitations

## 25.1 Severe class imbalance

`CL` represents only 2.64% of training data. No class weighting, focal loss, or resampling strategy was fully evaluated in the final pipeline.

## 25.2 Limited feature separation for `CL`

`CL` overlaps heavily with `C` and `D` across available predictors. The model may lack transplant-specific information needed for reliable separation.

## 25.3 Inferred meaning of structural missingness

`in_trial` is based on a clear block-completeness pattern, but its clinical meaning is inferred rather than independently confirmed.

## 25.4 Blend-selection optimism

Blend weights are optimized and scored on the same OOF predictions. The gain may be mildly optimistic.

## 25.5 Search-boundary diagnostics

The integer boundary check flags exact edge hits but is less sensitive to gradual edge pressure. CatBoost depth continued trending low after widening.

## 25.6 Seed-averaging uncertainty

Only a limited set of seeds was tested. The OOF/LB disagreement remains unresolved.

## 25.7 Limited interpretability

The final blend does not include a completed SHAP analysis or a formal feature-ablation study.

## 25.8 Limited distribution-shift analysis

Only selected univariate train/test comparisons were performed. Categorical and multivariate shift may remain.

## 25.9 Multiple execution environments

The final HTML export was executed locally on a MacBook Pro and records that local environment. The workflow also supports Google Colab/Drive, and earlier standard stages were tested in Kaggle-style CPU environments.

Scores should reproduce under fixed code, versions, data, and seeds, but runtime comparisons are not directly transferable across local, Colab, and Kaggle hardware. The recorded 5.58-hour timing belongs to the local final run.

## 25.10 Notebook-output consistency and naming

The final review found three documentation-level cleanup items:

- the Step 17 reproduction print statement incorrectly says Step 12 writes the final CSV;
- Experiment 5 is still labeled “final submission” in one generated table even though Experiment 10 superseded it;
- equivalent tuned and seed-averaged submissions are written under multiple filenames.

These do not change model predictions, but correcting them would make the notebook easier to audit.

## 25.11 Educational context

This is a competition project, not a clinically validated prediction system. It should not be used for medical decisions.

---

# 26. Future work

Future work should be prioritized by the evidence from error analysis rather than by adding complexity indiscriminately.

## Priority 1 — Improve the rare `CL` class

Potential experiments:

- class-weighted objectives;
- probability-aware cost-sensitive learning;
- focal-style loss where supported;
- carefully controlled oversampling;
- transplant-specific feature construction;
- hierarchical modeling, such as separating `C` versus adverse outcome before distinguishing `CL` and `D`.

Every method should still be evaluated with multiclass Log Loss, because improving recall can worsen probability calibration.

## Priority 2 — Formal uncertainty in model comparisons

Use:

- paired per-row Log Loss differences;
- bootstrap confidence intervals;
- repeated nested cross-validation;
- more seed sets.

This would provide stronger evidence when score differences are only a few thousandths.

## Priority 3 — Controlled feature ablations

Measure the isolated effect of:

- `in_trial`;
- missing indicators;
- high-missing laboratory columns;
- treating `Clinical_Stage` as categorical versus ordinal;
- explicit transformed or interaction features.

## Priority 4 — Calibration analysis

Possible methods:

- temperature scaling;
- isotonic regression where sample size permits;
- classwise calibration diagnostics;
- calibration evaluated only through nested or held-out predictions to avoid optimism.

## Priority 5 — Interpretability

Use SHAP or permutation importance to examine:

- which features influence `D` predictions;
- which features distinguish the few correctly identified `CL` cases;
- whether missingness proxies dominate the model unexpectedly.

## Priority 6 — Constrained blend optimization

Compare Nelder–Mead with:

- SLSQP under explicit `weights ≥ 0` and `sum(weights)=1` constraints;
- softmax-parameterized weights;
- nested blend-weight fitting to reduce selection bias.

## Priority 7 — Alternative tabular models

Models such as Random Forest, HistGradientBoosting, TabPFN, TabNet, or modern deep tabular architectures could be explored. They should be added only with a clear hypothesis and evaluated under the same OOF framework.

---

# 27. Reproduction guide

## 27.1 Required files

Obtain the competition data from the authorized AIO source:

```text
aio26_train.csv
aio26_test.csv
```

The public repository should not redistribute the data unless the organizer explicitly permits it.

## 27.2 Environment

Final recorded environment:

```text
Python           3.13.7
NumPy            2.3.3
pandas           2.3.3
scikit-learn     1.7.2
SciPy            1.16.2
XGBoost          3.3.0
LightGBM         4.7.0
CatBoost         1.2.10
```

Install repository dependencies with:

```bash
python -m pip install -r requirements.txt
```

On macOS, XGBoost and LightGBM may require OpenMP support such as Homebrew `libomp`.

## 27.3 Run order

1. Place `aio26_train.csv` and `aio26_test.csv` at the paths expected by the final notebook.
2. Open the `.ipynb` notebook corresponding to the final export `Kaggle1_Uyen8.html`. In the public repository, use a descriptive filename such as `notebooks/hepatic_risk_outcome_prediction.ipynb`.
3. Restart the kernel to clear stale state.
4. Select one path configuration—Colab/Drive or local execution—and leave the other commented out.
5. Run all cells from top to bottom.
6. Confirm that every expected model appears in the final `RESULTS` check.
7. Confirm that Step 16 writes `submission_blend_optimised_v3.csv`.
8. Verify that the displayed OOF score is 0.37010 and the blend weights match the documented values.

The `.ipynb`, exported `.html`, and this Markdown document should all be regenerated from the same final notebook version before publication.

## 27.4 Expected runtime

- Baseline training, tuned refits, comparisons, blends, and exports: approximately 2.2 minutes in the final recorded local run.
- Six Optuna studies: approximately 332.6 minutes.
- Recorded total excluding unretained seed-averaging wall time: 334.8 minutes, or 5.58 hours.
- Runtime will differ across local, Colab, and Kaggle hardware.

## 27.5 Expected final result

```text
Blend weights:
XGBoost  = 0.4308
LightGBM = 0.5683
CatBoost = 0.0008

OOF Log Loss = 0.37010
```

## 27.6 Submission verification

The final CSV should contain:

```text
10,000 rows
4 columns: id, Status_C, Status_CL, Status_D
no NaN values
probabilities summing to approximately 1 per row
```

---

# 28. Glossary

## Baseline

A simple reference model used to measure whether more complex methods provide meaningful improvement.

## Categorical feature

A variable containing groups or labels, such as patient sex or edema status.

## Class imbalance

A situation where some target classes have far fewer examples than others.

## Cross-validation

A repeated train/validation procedure that estimates model performance across several data partitions.

## Data leakage

Information from validation or test data influencing model fitting, causing overly optimistic evaluation.

## Early stopping

Stopping boosting when validation performance has not improved for a specified number of rounds.

## Feature engineering

Creating new predictors from existing data based on a hypothesis or observed structure.

## Fold

One subset in a cross-validation partition.

## Hyperparameter

A model setting selected before training, such as learning rate or tree depth.

## Log Loss

A probability-based metric that penalizes low probability assigned to the correct class, especially confident mistakes.

## Missing indicator

A binary variable recording whether another value was originally missing.

## Multiclass classification

Prediction among more than two discrete classes.

## OOF prediction

A prediction for a training row generated by a model that did not train on that row.

## Optuna

A hyperparameter-optimization framework used here with a TPE sampler.

## Probability calibration

The degree to which predicted confidence corresponds to observed correctness.

## Seed

A fixed number controlling reproducible pseudo-random operations such as data shuffling and model randomness.

## Seed averaging

Training under several seeds and averaging the resulting probability predictions.

## Stratification

Preserving target-class proportions when dividing data into folds.

## TPE sampler

An adaptive Optuna search method that uses outcomes from previous trials to guide new proposals.

---

## Final takeaway

This project is not presented as a single successful model run. It is presented as a complete experimental process:

```text
understand the metric
→ protect the meaning of the data
→ audit assumptions
→ connect EDA to decisions
→ validate honestly
→ compare simple and nonlinear models
→ tune systematically
→ document failed experiments
→ inspect class-specific errors
→ select the final model cautiously
→ preserve reproducibility evidence
```

The final score is one outcome of that process. The more important result is a reusable framework for building and evaluating future machine-learning projects on imperfect, imbalanced, real-world biomedical data.

---

## Data and medical-use notice

The competition data are not included in this public repository. They should be obtained only through the authorized competition or course channel and used according to the organizer's rules.

This project is an educational machine-learning exercise. It is not clinically validated and must not be used for diagnosis, treatment, or patient-care decisions.

---

# 29. Verified visual and output audit

This appendix records every visual output in the final HTML and the main printed outputs that determine the project narrative.

## 29.1 Visual-output inventory

| Figure | Notebook step | What is displayed | Main verified conclusion |
|---:|---|---|---|
| 1 | 4.1 | Target-class counts | `CL` is only 317 rows; stratification is necessary. |
| 2 | 4.2 | Missing percentage by column | Missingness is substantial and seven columns form a suspicious ~43% cluster. |
| 3 | 4.5 | Numerical distributions by class | Strong skew, overlap, and nonlinear class differences support tree ensembles. |
| 4 | 4.6 | Outcome composition within categorical levels | Multiple categorical levels have visibly different outcome composition. |
| 5 | 4.7 | Train/test numerical overlays | No severe univariate train/test shift is visible in the inspected variables. |
| 6 | 11 | OOF comparison and per-fold lines | Boosting strongly beats the linear baseline; XGBoost and LightGBM are near-ties. |
| 7 | 14.1 | Correlation heatmap for `P(Status=D)` | Correlations are 0.9911–0.9969, so ensemble diversity is limited but not zero. |
| 8 | 15.1 | Default XGBoost confusion matrices | `CL` recall is 0.158 and most `CL` rows become `C`. |
| 9 | 15.1 | Tuned LightGBM confusion matrices | `CL` recall rises to 0.196. |
| 10 | 15.1 | Final-blend confusion matrices | Hard-label behavior is similar to tuned LightGBM; `CL` remains difficult. |
| 11 | 15.3 | XGBoost confidence histogram and reliability curve | Correct predictions are more confident; low-confidence bins show some overconfidence. |
| 12 | 15.3 | Tuned LightGBM confidence histogram and reliability curve | Calibration pattern is nearly unchanged after tuning. |
| 13 | 15.3 | Final-blend confidence histogram and reliability curve | Blending improves Log Loss without a dramatic global calibration-shape change. |

## 29.2 Major printed-output audit

| Output | Exact verified value | What it establishes |
|---|---:|---|
| Training shape | `12,000 × 20` | One ID, 18 raw predictors, and `Status`. |
| Test shape | `10,000 × 19` | One ID and 18 raw predictors; no target. |
| Feature matrix | `12,000 × 19` | One engineered proxy was added after removing ID and target. |
| Shared validation | 5 folds of 9,600 train / 2,400 validation | Every model is evaluated under the same partition logic. |
| Logistic Regression OOF | `0.45596` | Linear baseline. |
| XGBoost OOF | `0.37412` | Best default standalone model. |
| LightGBM OOF | `0.37464` | Near-tie with XGBoost. |
| CatBoost OOF | `0.38347` | Stronger than baseline but weaker than XGB/LGBM. |
| Equal default blend | `0.37431` | Simple averaging does not help. |
| Optimized default blend | `0.37330` | Unequal weighting gives a modest gain. |
| Tuned XGBoost OOF | `0.37149` | Optuna improves XGBoost. |
| Tuned LightGBM OOF | `0.37094` | Strongest tuned standalone model. |
| Tuned CatBoost OOF | `0.37952` | Tuning helps but it remains weaker. |
| Tuned blend OOF | `0.37010` | Selected reliable ensemble. |
| Tuned blend Public LB | `0.38042` | Hidden-label evidence moves in the same direction as tuning. |
| Seed-averaged blend OOF | `0.36713` | Best local OOF score. |
| Seed-averaged Public LB | `0.38067` | Slightly worse than the simpler tuned blend; evidence disagrees. |
| Optuna runtime recorded | `334.8 min` | About 5.58 hours, excluding seed-averaging wall time. |
| Final artifact consistency | 11 expected result entries, 11 OOF arrays, 11 summary entries | Evidence files agree with the in-memory registry. |

## 29.3 Exact fold scores

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Overall OOF |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.46607 | 0.45791 | 0.46225 | 0.43779 | 0.45580 | 0.45596 |
| XGBoost | 0.37690 | 0.37036 | 0.37593 | 0.36208 | 0.38531 | 0.37412 |
| LightGBM | 0.37721 | 0.37117 | 0.37736 | 0.36234 | 0.38514 | 0.37464 |
| CatBoost | 0.38726 | 0.37929 | 0.38791 | 0.37240 | 0.39050 | 0.38347 |
| XGBoost tuned | 0.37289 | 0.36834 | 0.37178 | 0.36164 | 0.38282 | 0.37149 |
| LightGBM tuned | 0.37396 | 0.36840 | 0.37250 | 0.35897 | 0.38088 | 0.37094 |
| CatBoost tuned | 0.37959 | 0.37721 | 0.38604 | 0.36864 | 0.38612 | 0.37952 |

Fold 4 is consistently easier and fold 5 is consistently difficult for the boosting models. The repeated pattern is an argument for shared data difficulty, not merely random model-specific fluctuation.

## 29.4 Submission-preview audit

The HTML previews both the seed-averaged candidate and the selected tuned blend. For test ID `15000`:

| Candidate | `Status_C` | `Status_CL` | `Status_D` |
|---|---:|---:|---:|
| Seed-averaged | 0.894176 | 0.013582 | 0.092242 |
| Selected tuned blend | 0.892821 | 0.012067 | 0.095112 |

For test ID `15004`:

| Candidate | `Status_C` | `Status_CL` | `Status_D` |
|---|---:|---:|---:|
| Seed-averaged | 0.314830 | 0.005378 | 0.679791 |
| Selected tuned blend | 0.305815 | 0.006424 | 0.687761 |

These examples show that the two candidates are similar but not identical. Small probability reallocations across 10,000 rows are sufficient to change Log Loss by several ten-thousandths.

## 29.5 Internal notebook inconsistencies resolved in this documentation

The final HTML is executable evidence, but some comments and labels preserve earlier project states. This document resolves them explicitly:

1. **Experiment 5 is not the final submission.** It was the best default-model blend and was later superseded by Experiment 10.
2. **Step 13's immediate printout says “consider using” seed averaging.** The final rejection occurs only after Public LB evidence is incorporated in Steps 14 and 17.
3. **The final CSV is physically written in Step 16.** A Step 17 reproduction sentence saying the last line of Step 12 writes it is outdated.
4. **`submission_blend_tuned.csv` and `submission_blend_optimised_v3.csv` contain the same tuned-blend probability logic under different names.** The latter is the selected competition filename.
5. **Two seed-average filenames are produced:** one with a hyphen and one with an underscore. This is naming duplication, not two different models.
6. **`Clinical_Stage` is not purely nominal.** The notebook treats it as categorical to avoid assuming equal linear spacing between stages.
7. **`in_trial` is a proxy name.** The data establish block completeness, not a verified causal statement about enrollment.

---

# 30. Notebook cell map

The final HTML contains 73 cells. The map below connects the rendered notebook to this document without reproducing every line of code.

| HTML cells | Content | Outputs reviewed |
|---|---|---|
| 2–5 | Problem statement, pipeline structure, imports, and configuration | Package versions; seed, folds, and class order. |
| 6–8 | Data loading and initial description | Train/test shapes, first five rows, class counts, data types, and number of columns with missing values. |
| 9–10 | Structural audit | Target classes, feature alignment, duplicate IDs, target mapping, and unseen categories. |
| 11–18 | EDA | Five charts plus missingness-by-class and joint-block output. |
| 19–20 | Feature construction | Matrix shapes, target shape, exact numerical/categorical feature lists, and `in_trial` counts. |
| 21–24 | Validation framework | Exact fold sizes and class proportions; confirmation that the generic CV driver is ready. |
| 25–26 | Logistic Regression | Fold losses and overall OOF score. |
| 27–28 | XGBoost | Hyperparameters, fold losses, OOF score, and runtime. |
| 29–30 | LightGBM | Hyperparameters, fold losses, OOF score, and runtime. |
| 31–32 | CatBoost | Native categorical workflow, fold losses, OOF score, and runtime. |
| 33–35 | Default comparison | Ranked table, baseline improvements, performance chart, and best-default-model selection. |
| 36–45 | Two-round Optuna tuning | Scout warnings, final parameters, refit scores, tuned blend weights, and result registration. |
| 46–50 | Seed averaging | Per-seed fold outputs, averaged scores, re-blended OOF score, and registration. |
| 51–55 | Default blending and all-blend comparison | Correlation heatmap, equal blend, optimized default weights, and OOF/LB disagreement note. |
| 56–62 | Error and calibration analysis | Three confusion-matrix figures, per-class table, three confidence/reliability figures, and confidence summaries. |
| 63–68 | Submission generation | Eleven automatically generated submission files and previews of the two leading candidates. |
| 69–72 | Evidence and reproducibility | Experiment table, environment, tuning runtime, final parameters, artifact checks, final ranking, and file inventory. |
| 73 | Report mapping appendix | Links notebook evidence to technical-report requirements. |

This map confirms that the pipeline narrative is not inferred from model names alone. It is grounded in the actual executed code and outputs of the final notebook.

Thank you for visiting — and for being part of my journey 🌱
