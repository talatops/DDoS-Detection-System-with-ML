### ML Preprocessing & Model Training Overview

This document explains how the DDoS detection system preprocesses data and trains machine‑learning models, with a focus on the Random Forest (RF) and Deep Neural Network (DNN) models. It also summarizes windowing, row sizes, and k‑fold cross‑validation, and lists the key files involved.

---

### 1. End‑to‑End Pipeline Overview

At a high level, the offline training flow is:

1. **Load CSV flow data** from one or more dataset directories.
2. **Extract tabular features** from each row (flow) and, in batches of rows, compute entropy/uniqueness statistics.
3. **Preprocess features** using a reusable `FeaturePreprocessor` (outlier clipping, optional scaling/feature selection).
4. **Split into train and test sets** with a configurable test fraction.
5. **Train models** (RF, GBDT, and/or DNN) on the training data.
6. **Evaluate models** using a held‑out test set and k‑fold cross‑validation on the training set.
7. **Select a best model**, save trained artifacts and the preprocessor, and write a `model_manifest.json` plus text/JSON reports.

The main multi‑model training entrypoint is:
- `src/ml/train_models.py`

An older/auxiliary training script that also defines the core feature extraction is:
- `src/ml/train_ml.py`

At runtime (online detection), packet‑level statistics are aggregated in fixed‑length time windows by:
- `src/ingest/window_manager.h`
- `src/ingest/window_manager.cpp`

The preprocessing logic shared across models is implemented in:
- `src/ml/preprocessor.py`

The DNN wrapper that makes PyTorch models look like scikit‑learn classifiers is:
- `src/ml/model_wrappers.py`

Model metadata and performance summaries are stored in:
- `models/model_manifest.json`

---

### 2. Data Sources, Rows, and Windowing

#### 2.1 CSV loading and labels

File: `src/ml/train_ml.py`

- Function `load_training_data(csv_dir="data/caida-ddos2007", max_rows=None)`:
  - Recursively finds all `*.csv` files under the given directory.
  - Loads each CSV (optionally limiting rows via `max_rows`) and ensures there is a binary label column:
    - If a column whose name (case‑insensitive) matches `"label"` exists, it is mapped to:
      - `0` for `BENIGN`
      - `1` for all non‑BENIGN labels.
    - If no label column exists, the script infers label `1` (attack) or `0` (benign) from the filename using attack‑related keywords (e.g., `ddos`, `udp`, `syn`, etc.).
  - All per‑file DataFrames are concatenated into a single combined DataFrame `df` with:
    - One **row per network flow** (or record) in the original CSV.
    - A binary `label` column used for supervised training.

`src/ml/train_models.py` uses this same loader via:
- `train_ml.load_training_data(dataset, max_rows=limit)` inside `combine_datasets()`.
- The `--datasets` CLI flag specifies one or more dataset directories.
- The `--max-csv-rows` flag maps to the `max_rows` argument to cap per‑file rows for speed.

#### 2.2 Row size and window size in feature extraction

File: `src/ml/train_ml.py`

The core feature builder is:
- `extract_features_from_csv(df)`

Key concepts:

- **Row size**:
  - Each row in `df` corresponds to one flow/record from the CSV.
  - Features are built per row, using columns such as packet counts, byte counts, durations, and flow‑level statistics.

- **Statistical row windows (for entropy)**:
  - To approximate traffic diversity over time or over batches of flows, the script processes the DataFrame in **fixed windows of rows**:
    - `window_size = 1000`  →  each window contains up to **1000 rows (flows)**.
    - The number of windows is `n_windows = len(df) // window_size`, with a final partial window if needed.
  - For each window of rows, it calculates:
    - **Source IP entropy**, **destination IP entropy**
    - **Source port entropy**, **destination port entropy**
    - **Protocol entropy**
    - **Counts of unique src/dst IPs and ports**
    - **Fractions of traffic contributed by the top 10 source and destination IPs**
  - These window‑level statistics are then written back to **all rows in that window**, so each row carries both per‑flow and per‑window context.

Concretely:
- **Row size**: one flow per row.
- **Entropy window size**: fixed **1000 rows** per window in `extract_features_from_csv`.

#### 2.3 Online time windows (`WindowManager`)

Files: `src/ingest/window_manager.h`, `src/ingest/window_manager.cpp`

For live packet ingest, the system uses time‑based windows:

- Class: `WindowManager`
  - Constructor: `WindowManager(uint32_t window_size_sec = 1);`
  - Internal field: `window_size_us_ = window_size_sec * 1,000,000ULL;`
  - Default **time window size**: **1 second**.

`WindowManager` accepts individual `PacketInfo` messages via `addPacket()` and maintains a `WindowStats` struct per active window that tracks:
- Histograms:
  - `src_ip_counts`, `dst_ip_counts`
  - `src_port_counts`, `dst_port_counts`
  - `packet_size_counts`
  - `protocol_counts`
- Aggregated stats:
  - `total_packets`, `total_bytes`
  - `unique_src_ips`, `unique_dst_ips`
  - `flow_count`, `tcp_packets`, `udp_packets`
  - `syn_packets`, `fin_packets`, `rst_packets`, `ack_packets`
- Window timing:
  - `window_start_us`, `window_end_us`

Windows are closed when:
- `checkWindow(current_time_us)` observes that
  - `elapsed >= window_size_us_`
  - then `closeWindow()` finalizes `WindowStats`, triggers a callback to downstream code, and resets for the next window.

Summary:
- **Offline training** uses **row windows of 1000 flows** for entropy and uniqueness.
- **Online detection** uses **time windows of 1 second by default** for packet aggregation.

---

### 3. Feature Extraction and Preprocessing

#### 3.1 Feature extraction from CSV flows

File: `src/ml/train_ml.py`

`extract_features_from_csv(df)` constructs a feature matrix `df_features` indexed by the same rows as the input `df`. Key types of features:

- **Basic counts and sizes**:
  - `total_packets` = forward + backward packet counts.
  - `total_bytes` = forward + backward byte counts.
  - `avg_packet_size` = `total_bytes / total_packets` (with safe handling of zeros).
  - `packet_size_mean` and `packet_size_std` from forward/backward packet length stats.

- **Flow duration and rates**:
  - `flow_duration`
  - `flow_bytes_per_sec`
  - `flow_packets_per_sec`

- **TCP flag counts** (if columns present):
  - `syn_flag_count`
  - `fin_flag_count`
  - `rst_flag_count`

- **Entropy and uniqueness per 1000‑row window**:
  - Entropy of:
    - `src_ip`, `dst_ip`
    - `src_port`, `dst_port`
    - `protocol`
  - Uniqueness:
    - `unique_src_ips`, `unique_dst_ips`
    - `unique_src_ports`, `unique_dst_ports`
  - Concentration:
    - `top10_src_ip_fraction`
    - `top10_dst_ip_fraction`

- **Additional derived features**:
  - `packet_size_entropy` approximated via coefficient of variation:
    - `packet_size_std / (packet_size_mean + 1e-6)`
  - `flow_count` (set to 1 per row; useful for aggregation or consistency).

All missing values are filled with zeros, so models do not see NaNs.

#### 3.2 Preprocessing pipeline (`FeaturePreprocessor`)

File: `src/ml/preprocessor.py`

Class: `FeaturePreprocessor`

Core responsibilities:
- Optional **outlier removal** (clipping).
- Optional **scaling**.
- Optional **feature selection**.
- Persistence via `joblib`.

Key configuration options:
- `use_scaling` (bool; default `True` in class, but overridden per script).
- `use_outlier_removal` (bool; default `True`).
- `use_feature_selection` (bool; default `False`).
- `n_features` (int or `None`): number of features when feature selection is enabled.

Outlier removal:
- The preprocessor clips each feature to precomputed percentiles:
  - Lower percentile: **1st percentile** (`outlier_lower = 0.01`).
  - Upper percentile: **99th percentile** (`outlier_upper = 0.99`).
- On `fit()`:
  - It computes lower and upper bounds per feature over the training data.
- On `transform()`:
  - It clips each feature value into `[lower_bounds_, upper_bounds_]`.

Scaling:
- If `use_scaling` is `True`, it uses a `StandardScaler` (zero mean, unit variance) to normalize features.
- In the current multi‑model training script (`train_models.py`), scaling is turned **off** by default (see below), relying on tree‑based models’ robustness to unscaled features.

Feature selection:
- If `use_feature_selection` is `True` and `y` is provided:
  - It uses `SelectKBest(f_classif, k=n_features)` to select the top‑K features by ANOVA F‑score.
  - `n_features` defaults to `min(X.shape[1], 20)` when not specified, in the generic `FeaturePreprocessor`; in `train_models.py` it is passed as `min(24, X.shape[1])` but feature selection is off.

#### 3.3 How `train_models.py` configures preprocessing

File: `src/ml/train_models.py`

Function: `preprocess_features(df: pd.DataFrame) -> (X_array, y, preprocessor)`

- Extracts features via `train_ml.extract_features_from_csv(df)` (see above).
- Ensures `X` is a DataFrame if needed, then constructs:
  - `FeaturePreprocessor(`  
    `    use_scaling=False,`  
    `    use_outlier_removal=True,`  
    `    use_feature_selection=False,`  
    `    n_features=min(24, X.shape[1]),`  
    `)`
- Applies `fit_transform(X, y)` to produce `X_processed`.
- Converts to a NumPy `float32` array and replaces NaNs and ±∞ with 0.

So for current multi‑model training:
- **Outlier clipping**: **enabled** (1st–99th percentile).
- **Scaling**: **disabled**.
- **Feature selection**: **disabled** (even though `n_features` is set).

#### 3.4 How `train_ml.py` configures preprocessing

File: `src/ml/train_ml.py`

In `main()`:
- After feature extraction (`X = extract_features_from_csv(df)`), it sets:
  - `n_features_to_select = min(15, max(5, len(X.columns) - 5))`
  - Builds a `FeaturePreprocessor` with:
    - `use_scaling=False`
    - `use_outlier_removal=True`
    - `use_feature_selection=False`
    - `n_features=n_features_to_select`
- It then `fit_transform`s this preprocessor on `X` and `y`, similarly to `train_models.py`.

In practice, both scripts currently use:
- **Outlier removal on**, **scaling off**, **feature selection off**.

---

### 4. Model Training Configuration (RF and DNN)

`src/ml/train_models.py` is the main entrypoint for training multiple models together and producing a shared manifest and reports.

CLI arguments (subset relevant here):
- `--datasets`: one or more dataset directories (default: `data/caida-ddos2007`).
- `--models`: list of models to train; choices: `["rf", "gbdt", "dnn"]` (default: all three).
- `--test-size`: fraction of data used as test split (default: `0.3`).
- `--kfolds`: number of folds for cross‑validation (default: `5`).
- `--max-csv-rows`: limit on per‑file CSV rows (`-1` means all).
- `--balance-data` and `--max-ratio`: optional down‑sampling of majority class (e.g., benign) to reduce class imbalance.

After preprocessing:
- The script performs a train/test split:
  - `train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)`.
- It saves the fitted `FeaturePreprocessor` to `models/preprocessor.joblib`.
- It then trains each requested model and evaluates it.

#### 4.1 Random Forest (RF)

File: `src/ml/train_models.py`

Function: `train_random_forest(X_train, y_train) -> RandomForestClassifier`

Hyperparameters:
- `n_estimators = 256`
- `max_depth = 18`
- `n_jobs = -1` (use all CPU cores)
- `class_weight = "balanced"` (compensates for label imbalance)
- `random_state = 42`

Training:
- `model.fit(X_train, y_train)`

Evaluation:
- After training, `evaluate_classifier()` is used:
  - Computes train and test accuracies.
  - Performs k‑fold cross‑validation on training data (see Section 5).
  - Calculates precision, recall, false positive rate (FPR), ROC AUC, PR AUC, confusion matrix, and a text classification report.

Artifact and manifest entry:
- The trained model is saved to:
  - `models/rf_model.joblib`
- The manifest entry (in `models/model_manifest.json`) records:
  - `name: "rf"`
  - `type: "random_forest"`
  - `path: "models/rf_model.joblib"`
  - `preprocessor: "models/preprocessor.joblib"`
  - Evaluation metrics: `recall`, `false_positive_rate`, `roc_auc`.

#### 4.2 DNN (Torch MLP)

Files: `src/ml/train_models.py`, `src/ml/model_wrappers.py`

Function: `train_dnn(X_train, y_train, epochs: int = 5, batch_size: int = 512)`

Architecture:
- Input layer: dimension same as number of features.
- Hidden layers: `[128, 64, 32]` with ReLU activations.
- Output layer: single linear neuron (logit for binary classification).

Training configuration:
- Optimizer: `Adam` with learning rate `1e-3`.
- Loss: `BCEWithLogitsLoss` (suitable for binary classification with logits).
- Default epochs: `5`.
- Batch size: `512`.
- Device:
  - Prefers GPU if available (`torch.cuda.is_available()`), otherwise falls back to CPU.

Wrapping for inference:
- After training, the script:
  - Extracts the model `state_dict` to CPU.
  - Constructs a `TorchModelWrapper` with:
    - `input_dim`
    - `hidden_sizes`
    - `state_dict`
- `TorchModelWrapper` (from `src/ml/model_wrappers.py`) provides:
  - `predict_proba(features)`:
    - Applies the MLP and a sigmoid to obtain probabilities.
    - Returns a 2‑column array `[P(class=0), P(class=1)]` to mimic scikit‑learn.

This allows the DNN to be evaluated using the same `evaluate_classifier()` function as the RF.

Artifact and manifest entry:
- The wrapped DNN model is saved to:
  - `models/dnn_model.joblib`
- The manifest entry records:
  - `name: "dnn"`
  - `type: "torch_mlp"`
  - `path: "models/dnn_model.joblib"`
  - `preprocessor: "models/preprocessor.joblib"`
  - `imports`: `["preprocessor", "model_wrappers"]`
  - Evaluation metrics: `recall`, `false_positive_rate`, `roc_auc`.

---

### 5. Train/Test Split, K‑Folds, and Metrics

#### 5.1 Train/test split

In `src/ml/train_models.py`, the main script splits the data as:
- `train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)`

Defaults:
- **Test size**: `0.3` (30% of samples for the test set).
- **Stratified**: preserves label balance across train and test splits.

In `src/ml/train_ml.py`, an internal `train_model()` uses:
- `test_size=0.2` (20% test) for that standalone RF training.

#### 5.2 K‑fold cross‑validation (what it is and how it is used)

**Conceptual definition (what is k‑folds)**:

- K‑fold cross‑validation is a technique to estimate model performance more robustly:
  1. Split the training data into **k equal folds** (subsets).
  2. For each fold:
     - Train on the other `k-1` folds.
     - Validate on the held‑out fold.
  3. Aggregate the score (here, F1 score) across all k runs.
- This reduces variance compared to a single train/validation split and helps detect overfitting.

In this project:

- `src/ml/train_models.py`:
  - `evaluate_classifier()` receives `kfolds` (default `5`).
  - It sets:
    - `cv_splits = max(2, kfolds)`
    - `cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splits, scoring="f1")`
  - That is, **k‑folds = 5** by default, adjustable via the `--kfolds` CLI flag.

- `src/ml/train_ml.py`:
  - The legacy `train_model()` uses a fixed:
    - `cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')`
  - Here, **k‑folds = 5** is hard‑coded.

So:
- **Default k‑fold setting**: **5 folds** (k = 5) for cross‑validation in both scripts.

#### 5.3 Metrics computed

`evaluate_classifier()` in `src/ml/train_models.py` computes:
- **Train and test accuracy**.
- **Cross‑validation F1 statistics**:
  - `cv_f1_mean`
  - `cv_f1_std`
- **Confusion matrix components**:
  - True negatives (TN)
  - False positives (FP)
  - False negatives (FN)
  - True positives (TP)
- **Derived metrics**:
  - Precision
  - Recall
  - False positive rate (FPR)
- **Curve‑based metrics**:
  - ROC AUC (area under the ROC curve).
  - PR AUC (area under the precision‑recall curve).
- **Classification report**:
  - Human‑readable precision/recall/F1 per class.

`evaluate_on_dataset()` provides similar metrics for additional evaluation datasets.

These metrics are saved into reports under `reports/` and into `reports/training_metrics.json`.

---

### 6. Model Manifest and Selection Logic

File: `src/ml/train_models.py` and `models/model_manifest.json`

After training all requested models:
- `select_best_model(results)` ranks the models using the heuristic:
  - Score = `recall - 0.5 * false_positive_rate`
  - The model with the highest score is selected.

`save_manifest()` then:
- Reads any existing manifest to merge entries.
- Updates or inserts entries for the models just trained.
- Persists:
  - `generated`: timestamp.
  - `default_model`: historically selected default (preserved if present).
  - `selected_model`: the model chosen **in the current run** by the heuristic above.
  - `models`: list of per‑model objects containing:
    - Name, type, artifact path.
    - Preprocessor path.
    - Imports (for DNN).
    - Performance metrics (`recall`, `false_positive_rate`, `roc_auc`).

Current manifest snapshot (`models/model_manifest.json`):
- `default_model`: `"dnn"`
- `selected_model`: `"gbdt"`
- Models:
  - `"dnn"`:
    - `recall`: ~0.971
    - `false_positive_rate`: ~0.653
    - `roc_auc`: ~0.957
  - `"rf"`:
    - `recall`: ~0.986
    - `false_positive_rate`: ~0.031
    - `roc_auc`: ~0.998

Even though `gbdt` is the currently `selected_model` in this snapshot, the comparison you requested focuses on **RF vs DNN**, and those metrics show a very large performance gap in favor of RF.

---

### 7. RF vs DNN: Which Model Is Better and Why?

Based on `models/model_manifest.json` and the evaluation logic:

- **Random Forest (RF)**:
  - **Recall**: ~0.986 — detects almost all attacks.
  - **False positive rate (FPR)**: ~0.031 — very low; only about 3.1% of benign flows are misclassified as attacks.
  - **ROC AUC**: ~0.998 — near‑perfect discrimination between benign and attack.
  - Architecture and training:
    - Tree‑based, robust to unscaled numeric features.
    - Naturally handles non‑linear interactions and mixed‑scale features.
    - `class_weight="balanced"` helps compensate for label imbalance.

- **DNN (torch MLP)**:
  - **Recall**: ~0.971 — still high, but slightly lower than RF.
  - **False positive rate (FPR)**: ~0.653 — extremely high; more than 65% of benign flows are incorrectly flagged as attacks.
  - **ROC AUC**: ~0.957 — good but noticeably worse than RF.
  - Architecture:
    - Fully‑connected MLP with `[128, 64, 32]` hidden units and ReLU.
    - Trained for a relatively small number of epochs (default 5), with basic hyperparameters and no extensive tuning or regularization search.

#### 7.1 Practical implications

- In a production DDoS detection system:
  - **High recall** is important because we want to catch most attacks.
  - **Low false positive rate** is equally critical; otherwise, benign traffic is frequently blocked or flagged, causing operational issues.
- The RF model combines **higher recall** with an **orders‑of‑magnitude lower FPR** and a substantially better ROC AUC.
- The DNN’s very high FPR makes it unsuitable as‑is for deployment, despite its decent recall.

#### 7.2 Conclusion: which is better?

- **Random Forest (RF) is clearly the better model in this codebase right now**, because:
  - It has **better core metrics** (higher recall, much lower FPR, higher ROC AUC).
  - It is **simpler to train and operate** on tabular features, with fewer tuning knobs.
  - It is **less sensitive to preprocessing details** (e.g., lack of scaling) than the DNN.
- The **DNN** remains a useful experimental baseline, but to become competitive, it would need:
  - More extensive hyperparameter tuning (layers, dropout, learning rate, epochs).
  - Potentially more sophisticated preprocessing and/or regularization.

---

### 8. Files Involved and How to Run Training

#### 8.1 Key files

- `src/ml/train_models.py`
  - Main multi‑model training script.
  - Handles dataset loading, preprocessing, train/test split, training of RF/GBDT/DNN, evaluation, and manifest/report generation.

- `src/ml/train_ml.py`
  - Legacy/supplemental RF training script.
  - Defines `load_training_data()` and `extract_features_from_csv()`, which are reused by `train_models.py`.

- `src/ml/preprocessor.py`
  - Defines the `FeaturePreprocessor` class used to clip outliers, optionally scale features, and (optionally) perform feature selection.
  - Provides save/load utilities via `joblib`.

- `src/ml/model_wrappers.py`
  - Implements `TorchMLP` and `TorchModelWrapper` to expose a DNN with a `predict_proba()` interface compatible with scikit‑learn.

- `src/ingest/window_manager.h`, `src/ingest/window_manager.cpp`
  - Implement the online time‑windowing of packet statistics (default **1‑second windows**) for runtime detection.
  - Provide `WindowStats` aggregates that conceptually mirror some of the entropy/uniqueness features used offline.

- `models/model_manifest.json`
  - Persisted registry of trained models.
  - Stores per‑model paths, types, imports, and headline metrics.
  - Contains `default_model` and `selected_model` fields for downstream components (e.g., dashboard, inference engine).

#### 8.2 How to run training (RF and DNN focus)

Example: train RF and DNN on the default dataset with 5‑fold CV:

```bash
cd /home/talatfaheem/PDC/project
python3 src/ml/train_models.py \
  --datasets data/caida-ddos2007 \
  --models rf dnn \
  --kfolds 5 \
  --test-size 0.3
```

Key flags:
- `--datasets`: change this to point to your labeled CSV directories.
- `--models`: choose any subset of `rf`, `gbdt`, `dnn` (for this comparison, `rf dnn` is most relevant).
- `--kfolds`: controls the number of folds for cross‑validation on the training set (default **5**).
- `--max-csv-rows`: if set to a positive integer, limits per‑CSV rows for faster experimentation.
- `--balance-data` and `--max-ratio`: enable down‑sampling of the majority class to reduce imbalance.

After the run:
- Trained models and preprocessor are saved under `models/`.
- Reports and metrics are written under `reports/`.
- The model registry is updated in `models/model_manifest.json`, with the best model (by recall/FPR trade‑off) marked as `selected_model`.


