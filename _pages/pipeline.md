---
layout: single
title: "Detailed Project Pipeline"
permalink: /pipeline/
author_profile: true
---

# Detailed Project Pipeline — Hepatic Risk Outcome Prediction

> **AIO 2026 Kaggle Competition 1**  
> **Author:** Uyen Tran (Victoria Tran)  
> **Task:** Predict three possible hepatic outcomes from structured clinical records  
> **Evaluation metric:** Multiclass Log Loss  
> **Final reliable submission:** `submission_blend_optimised_v3.csv`  
> **Final reliable scores:** OOF Log Loss **0.37010** · Public Leaderboard **0.38042**

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
4. **Mixed data types.** The dataset contains numerical measurements and nominal categorical variables that require different preprocessing strategies.
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

## 3.3 Target distribution

| Class | Rows | Percentage |
|---|---:|---:|
| `C` | 8,151 | 67.92% |
| `CL` | 317 | 2.64% |
| `D` | 3,532 | 29.43% |

The `CL` class is extremely rare. A model can achieve high overall accuracy while almost never identifying `CL` correctly. This is why class-specific error analysis is necessary.

## 3.4 Missing data

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
| Categorical variables are nominal | Avoid ordinal label encoding |
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

Eleven predictors contain missing values. Two laboratory columns are missing in more than 55% of rows. Seven other columns have missing rates clustered near 43%.

### Interpretation

The similar rates among seven different columns are unlikely to be a coincidence. They suggest a shared data-collection mechanism.

### Decision

Do not drop high-missing columns automatically. First investigate whether missingness carries information.

---

## 9.3 Missingness by target class

### Question

Does the probability of a value being missing differ among `C`, `CL`, and `D`?

### Observation

Missing rates differ by outcome. For several variables, the spread between classes is meaningful, reaching approximately 11 percentage points for cholesterol and triglycerides.

A recurring pattern is:

- `D` often has the highest missingness;
- `CL` often has the lowest;
- `C` lies between them.

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

Approximate breakdown:

```text
0 of 7 missing → 6,737 rows (56.1%)
7 of 7 missing → 5,147 rows (42.9%)
partial cases  → about 1% combined
```

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

They are primarily **nominal**, meaning their values do not have a valid numerical ranking. Encoding treatment categories as 0, 1, and 2 would create an artificial order.

### Decision

- Use one-hot encoding for Logistic Regression.
- Use the boosting libraries' supported categorical mechanisms for XGBoost, LightGBM, and CatBoost.
- Treat `Clinical_Stage` as categorical in the current pipeline despite its numeric storage format.

---

## 9.7 Train/test distribution comparison

### Question

Are the training and test datasets drawn from visibly different distributions?

### Observation

Selected numerical distributions and summary statistics overlap closely. For example, bilirubin and albumin means are similar between train and test.

### Interpretation

No severe univariate distribution shift is evident in the inspected variables.

This does not prove that the two datasets are identical. Categorical or multivariate shift may still exist.

### Decision

Use cross-validation as the main model-selection criterion and treat the public leaderboard as a secondary, noisier signal.

---

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

This gives tree models one clear structural signal rather than requiring them to rediscover the same condition repeatedly across several missing columns.

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

## Final dimensions

After adding `in_trial`:

```text
X      → 12,000 rows × 19 predictors
X_test → 10,000 rows × 19 predictors
```

The 19 predictors include:

- 12 numerical features, including `in_trial`;
- 7 categorical features.

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

## 11.5 The `run_cv` driver

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

## 11.6 Fold score versus overall OOF score

The notebook records both:

- Log Loss for each validation fold;
- Log Loss on the complete 12,000-row OOF matrix.

The complete OOF score is the primary comparison value because it evaluates every training row together.

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

## Result

```text
OOF Log Loss = 0.45596
Fold SD      = 0.00975
Runtime      ≈ 1 second
```

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

## Result

```text
OOF Log Loss = 0.37412
Fold SD      = 0.00768
Runtime      ≈ 9–10 seconds
```

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

## Result

```text
OOF Log Loss = 0.37464
Fold SD      = 0.00758
Runtime      ≈ 11–12 seconds
```

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

## Result

```text
OOF Log Loss = 0.38347
Fold SD      = 0.00669
Runtime      ≈ 8 seconds
```

## Interpretation

CatBoost strongly outperforms the linear baseline but underperforms XGBoost and LightGBM.

This is a useful result because theoretical suitability does not guarantee empirical superiority. The categorical variables have low cardinality, and the dataset's strongest signal may be driven more by numerical thresholds and interactions than by CatBoost's specialized categorical machinery.

CatBoost nevertheless has the lowest fold standard deviation among the default boosting models, indicating stable behavior across the shared folds.

### Decision

Keep CatBoost as an experimental and ensemble candidate, but allow validation evidence to determine its final influence.

---

# 16. Step 11 — Model comparison

## Default-model results

| Rank | Model | OOF Log Loss | Fold SD | Interpretation |
|---:|---|---:|---:|---|
| 1 | XGBoost | 0.37412 | 0.00768 | Best default model |
| 2 | LightGBM | 0.37464 | 0.00758 | Near-tie with XGBoost |
| 3 | CatBoost | 0.38347 | 0.00669 | Stable but weaker |
| 4 | Logistic Regression | 0.45596 | 0.00975 | Baseline |

## Main findings

### Finding 1 — All boosting models beat the linear baseline

This confirms that nonlinear decision boundaries and feature interactions are important.

### Finding 2 — XGBoost and LightGBM are effectively tied

Their score difference is much smaller than the observed variation among folds.

### Finding 3 — CatBoost is useful but weaker

It should not receive equal influence simply because it is designed for categorical data.

### Finding 4 — Fold difficulty is shared

The models show similar patterns of easier and harder folds. This suggests that some folds contain intrinsically harder patient cases rather than one algorithm failing uniquely on a particular split.

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

## 18.6 Component results

| Model | Single-seed tuned OOF | Three-seed averaged OOF |
|---|---:|---:|
| XGBoost | 0.37149 | 0.36830 |
| LightGBM | 0.37094 | 0.36734 |
| CatBoost | 0.37952 | 0.37587 |

All three aggregate OOF scores improve.

## 18.7 Blend result

```text
Single-seed tuned blend OOF = 0.37010
Seed-averaged blend OOF     = 0.36713
```

At first glance, seed averaging appears superior.

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

## 20.2 Confusion matrix

A confusion matrix converts probability predictions into hard labels by selecting the highest-probability class.

For the original default XGBoost model, normalized results are approximately:

| True class | Predicted C | Predicted CL | Predicted D |
|---|---:|---:|---:|
| C | 0.94 | 0.00 | 0.06 |
| CL | 0.58 | 0.16 | 0.26 |
| D | 0.26 | 0.00 | 0.73 |

The dominant problem is clear: most true `CL` cases are predicted as `C` or `D`.

## 20.3 Class-specific metrics

| Class | Samples | Recall: before / tuned / blend | Mean Log Loss: before / tuned / blend |
|---|---:|---:|---:|
| C | 8,151 | 0.936 / 0.934 / 0.935 | 0.199 / 0.198 / 0.198 |
| CL | 317 | 0.158 / 0.196 / 0.192 | 2.496 / 2.486 / 2.441 |
| D | 3,532 | 0.731 / 0.735 / 0.735 | 0.588 / 0.581 / 0.581 |

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

This suggests that the limiting factor is not merely hyperparameter choice. The current features may not contain enough information to separate transplant cases reliably.

## 20.6 Confidence analysis

The notebook also compares confidence for correct and incorrect predictions.

For the default model:

```text
Mean confidence when correct   ≈ 0.890
Mean confidence when incorrect ≈ 0.718
```

The positive gap is useful: the model is generally less confident when it is wrong.

However, this statistic alone does not prove perfect calibration. A complete calibration assessment requires comparing predicted confidence with observed accuracy across probability bins and considering the low sample count in the rare class.

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

## Final selected file

```text
submission_blend_optimised_v3.csv
```

This file is generated from the tuned blend selected in Step 12.

The seed-averaged candidate is also generated for experimental comparison but is not the primary final solution.

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

## 22.5 Final consistency check

The notebook verifies that all expected entries are present in `RESULTS` before printing the final table.

This protects against a common notebook problem: restarting the kernel and running only some cells, leaving stale or incomplete state in memory.

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

## 25.9 Split computing environments

Most of the notebook ran on Kaggle CPU, while long Optuna and seed-averaging stages ran on a local MacBook Pro. Scores should reproduce under fixed versions and seeds, but runtime will differ.

## 25.10 Educational context

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

1. Place the data files at the paths expected by the notebook.
2. Open `notebooks/hepatic_risk_outcome_prediction.ipynb`.
3. Restart the kernel.
4. Run all cells from top to bottom.
5. Confirm that every expected model appears in the final `RESULTS` check.
6. Confirm that the final submission is `submission_blend_optimised_v3.csv`.

## 27.4 Expected runtime

- Standard training, comparison, blending, error analysis, and export: approximately 2–3 minutes in the recorded environment.
- Two-round Optuna tuning: approximately 5.5 hours on the original local MacBook Pro.
- Runtime will differ across hardware and shared Kaggle resources.

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
