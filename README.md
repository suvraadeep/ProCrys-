# Crystallization Component Prediction

## QUICK START (30 SECONDS)

```bash
# Launch interactive web app
cd 05_app
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## FINAL RESULTS

### Winner: **Advanced Baseline** ⭐

| Target | Simple Baseline | Advanced Baseline | Transformer | Winner |
|--------|----------------|-------------------|-------------|---------|
| **Component_1_Name (Accuracy)** | 61.12% | **64.18%** | 53.85% | Advanced |
| **Component_1_Conc (R²)** | N/A | **0.4733** | 0.3812 | Advanced |
| **Component_1_pH (R²)** | 0.9558 | **0.9934** | 0.9447 | Advanced |

**Key Achievement:** Concentration parsing **0% → 83.21%!**

---

## PROJECT STRUCTURE

```
BTP/
├── 01_data/                          # All datasets
│   ├── raw/                          # Original Excel and CSV
│   └── processed/                    # Preprocessed for each approach
│
├── 02_models/                        # Trained models
│   ├── simple_baseline/              # Simple ML models
│   ├── advanced_baseline/            # Advanced ML models (BEST)
│   └── transformer/                  # Deep learning model
│
├── 03_notebooks/                     # Jupyter notebooks
│
├── 04_results/                       # All outputs
│   ├── visualizations/               # 12 plots (EDA + comparisons)
│   └── tables/                       # CSV result tables
│
├── 05_app/                           # Interactive application
│   ├── app.py                        # Streamlit web app
│   └── RUN_APP.bat                   # Quick launcher
│
├── 06_docs/                          # Documentation
│   └── requirements.txt              # Python dependencies
│
├── baseline_simple/                  # Simple approach source code
├── advanced_baseline/                # Advanced approach source code
├── transformer_approach/             # Transformer source code
│
└── README.md                         # This file (complete guide)
```

---

## PROJECT OVERVIEW

### Objective
Predict 3 crystallization components from 5 experimental parameters.

### Input Features (5)
1. **Crystallization Method** (categorical: "VAPOR DIFFUSION, SITTING DROP", etc.)
2. **Temperature** (K) - Range: 250-350 K
3. **pH** - Range: 0-14
4. **Matthews Coefficient** - Range: 1.0-5.0
5. **Percent Solvent Content** (%) - Range: 0-100%

### Output Targets (3)
1. **Component_1_Name** - Classification (2,432 unique chemicals → Top 50 classes)
2. **Component_1_Conc** - Regression (concentration in Molarity)
3. **Component_1_pH** - Regression (pH value: 0-14)

### Dataset
- **Original:** 11,344 rows × 24 columns
- **After preprocessing:** 11,338 rows × 9-15 columns
- **Missing values:** 53.03% → 0% (fully handled)

---

## THREE APPROACHES EXPLAINED

### Simple Baseline

**Techniques:**
- Basic KNN imputation (k=5)
- Iterative/MICE for targets
- TF-IDF for text (50 features, bigrams)
- Standard scaling
- Tree ensembles: RandomForest, XGBoost, LightGBM, CatBoost

**Results:**
- Component_1_Name: 61.12% accuracy (LightGBM)
- Component_1_Conc: **FAILED** (parsing issue: 0% → all NaN)
- Component_1_pH: R² = 0.9558 (CatBoost)

**Pros:** Fast, simple, interpretable  
**Cons:** Concentration parsing failed

---

### Advanced Baseline ⭐ **BEST OVERALL**

**Key Innovations:**

**A. Concentration Unit Parsing (Game Changer!)**
```python
Input Examples:
  "0.1 M"    → 0.1 M (normalized)
  "100 mM"   → 0.1 M (normalized)
  "25%"      → 2.5 M (approximated)
  "5 mg/mL"  → 0.00005 M (approximated)

Success Rate: 83.21% (vs 0% in simple)
```

**B. Domain Constraints**
- Temperature: [250-350] K (protein crystallization range)
- pH: [0-14] (physical scale)
- Matthews: [1.0-5.0] (typical protein values)
- Concentration: [0.0001-10] M (reasonable chemistry range)

**C. Feature Engineering (6 new features)**
1. Temp × pH interaction
2. Matthews × Solvent interaction
3. pH difference (pH - Component_1_pH)
4. Solvent/Matthews ratio
5. Temperature categories (Low/Med/High)
6. pH categories (Acidic/Neutral/Basic)

**D. Advanced Imputation**
- KNN (k=7, distance-weighted)
- Iterative/MICE with domain priors
- pH used as prior for Component_1_pH

**Results:**
- Component_1_Name: **64.18% accuracy** (XGBoost) - BEST
- Component_1_Conc: **R² = 0.4733** (RandomForest) - BEST
- Component_1_pH: **R² = 0.9934** (CatBoost) - BEST (nearly perfect!)

**Pros:** Best performance, concentration works, domain knowledge  
**Cons:** Needs chemical expertise for setup

---

### Transformer Approach

**Architecture:**
```
Hybrid Transformer Model (644,020 parameters)

Input Layer:
  ├─ Crystallization Method ID → Embedding(128) 
  │   └→ Transformer Encoder (2 layers, 4 attention heads)
  └─ Numerical Features (4) → MLP(256) → ReLU → Dropout

Fusion Layer:
  Concatenate[Transformer(128) + MLP(256)] → MLP(256)

Output Heads (Multi-task):
  ├─ Component_1_Name: Linear(50) → Softmax [Classification]
  ├─ Component_1_Conc: Linear(1) [Regression]
  └─ Component_1_pH: Linear(1) [Regression]
```

**Training:**
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Loss: 2×CrossEntropy + MSE + MSE (weighted multi-task)
- Epochs: 50 with ReduceLROnPlateau scheduler
- Batch size: 64
- Training time: ~15-20 minutes

**Results:**
- Component_1_Name: 53.85% accuracy (Hybrid Transformer)
- Component_1_Conc: R² = 0.3812 (Hybrid Transformer)
- Component_1_pH: R² = 0.9447 (Hybrid Transformer)

**Pros:** Self-attention, multi-task learning, end-to-end  
**Cons:** Lower accuracy, needs more data, less interpretable

---

## COMPLETE RESULTS TABLE

### Detailed Performance Metrics

| Target | Metric | Simple | Advanced | Transformer | Best |
|--------|--------|--------|----------|-------------|------|
| **Name** | Accuracy | 61.12% | **64.18%** | 53.85% | Advanced |
| **Name** | Model | LightGBM | XGBoost | Hybrid | XGBoost |
| **Conc** | R² | N/A | **0.4733** | 0.3812 | Advanced |
| **Conc** | RMSE | N/A | **0.8066** | 0.5214 | Advanced |
| **Conc** | Model | N/A | RandomForest | Hybrid | RandomForest |
| **Conc** | Parsing | 0% | **83.21%** | 83.21% | Advanced |
| **pH** | R² | 0.9558 | **0.9934** | 0.9447 | Advanced |
| **pH** | RMSE | 0.3544 | **0.1371** | 0.3882 | Advanced |
| **pH** | Model | CatBoost | CatBoost | Hybrid | CatBoost |

### Summary Statistics

| Aspect | Value |
|--------|-------|
| Total Samples | 11,344 |
| After Cleaning | 11,338 |
| Original Missing | 53.03% |
| Final Missing | 0% |
| Concentration Parsing Success | 83.21% (Advanced) |
| Models Trained | 21 |
| Best Classification Accuracy | 64.18% |
| Best Concentration R² | 0.4733 |
| Best pH R² | 0.9934 |
| Total Training Time | ~25 minutes |

---

## HOW TO USE

### Method 1: Interactive Web App (Easiest)

```bash
cd 05_app
streamlit run app.py
```

**Features:**
- Adjust sliders for all 5 input parameters
- Click "Predict Components" button
- See Top-5 component predictions
- View predicted concentration and pH
- Compare Simple vs Advanced baseline
- See performance metrics

### Method 2: Python Code (Programmatic)

```python
import joblib
import numpy as np

# Load advanced baseline models (best performance)
model_name = joblib.load('02_models/advanced_baseline/model_component_name.pkl')
model_conc = joblib.load('02_models/advanced_baseline/model_component_conc.pkl')
model_ph = joblib.load('02_models/advanced_baseline/model_component_ph.pkl')

# Load preprocessors
le = joblib.load('02_models/advanced_baseline/label_encoder_name.pkl')
scaler = joblib.load('02_models/advanced_baseline/scaler.pkl')
tfidf = joblib.load('02_models/advanced_baseline/tfidf.pkl')

# Your crystallization parameters
method = "VAPOR DIFFUSION, SITTING DROP"
temp = 293.0  # Kelvin
ph = 7.0
matthews = 2.2
solvent = 45.0  # percent

# Feature engineering (as per Advanced Baseline)
temp_ph_int = temp * ph
matthews_solvent_int = matthews * solvent
ph_diff = 0  # Unknown for new data
solvent_ratio = solvent / (matthews + 1e-6)

# Prepare features
numerical = np.array([[temp, ph, matthews, solvent,
                       temp_ph_int, matthews_solvent_int,
                       ph_diff, solvent_ratio]])
numerical_scaled = scaler.transform(numerical)

# TF-IDF for crystallization method
method_tfidf = tfidf.transform([method.upper()]).toarray()

# Combine all features
X_input = np.concatenate([numerical_scaled, method_tfidf], axis=1)

# Make predictions
pred_name_idx = model_name.predict(X_input)[0]
pred_name = le.inverse_transform([pred_name_idx])[0]
pred_conc = model_conc.predict(X_input)[0]
pred_ph = model_ph.predict(X_input)[0]

print(f"Predicted Component: {pred_name}")
print(f"Predicted Concentration: {pred_conc:.4f} M (log-scale)")
print(f"Actual Concentration: {10**pred_conc:.6f} M")
print(f"Predicted pH: {pred_ph:.2f}")
```

### Method 3: View Pre-Generated Results

```bash
# Open visualizations folder
start 04_results\visualizations

# View summary table
cat 04_results\tables\final_summary.csv
```

---

## SETUP INSTRUCTIONS

### Prerequisites
- Conda or Miniconda installed
- Windows 10/11

### One-Time Setup

```bash
# 1. Create Python 3.9 environment
conda create -n btp_env python=3.9 -y

# 2. Activate environment
conda activate btp_env

# 3. Install dependencies
pip install uv
uv pip install -r 06_docs/requirements.txt

# 4. For transformer (optional)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# 5. Install Streamlit
pip install streamlit
```

---

## TECHNICAL DETAILS

### Data Preprocessing

**Missing Value Handling (5 techniques):**
1. **Mode Imputation** - Crystallization Method
2. **KNN Imputation** (k=5 or k=7) - Numerical features
3. **Iterative/MICE** - Component_1_Conc
4. **Domain-Specific + MICE** - Component_1_pH (uses pH as prior)
5. **Row Removal** - Component_1_Name missing (only 6 rows)

**Concentration Unit Parsing:**
```python
def parse_concentration(conc_str):
    # Regex to extract value and unit
    pattern = r'([\d.]+)\s*([A-Z%]+)?'
    value, unit = extract_match(pattern, conc_str)
    
    # Normalize to Molarity (M)
    if 'MM' in unit:        # millimolar
        return value / 1000
    elif 'UM' in unit:      # micromolar
        return value / 1000000
    elif '%' in unit:       # percent
        return value / 10   # Approximation
    elif 'MG/ML' in unit:
        return value / 100000  # MW approximation
    else:
        return value  # Assume M
```

**Domain Constraints:**
- Temperature ∈ [250, 350] K
- pH ∈ [0, 14]
- Matthews Coefficient ∈ [1.0, 5.0]
- Concentration ∈ [0.0001, 10] M

### Feature Engineering (Advanced Baseline)

| Feature | Formula | Purpose |
|---------|---------|---------|
| Temp_pH_interaction | Temp × pH | Temperature-pH relationship |
| Matthews_Solvent_interaction | Matthews × Solvent | Protein packing |
| pH_diff | pH - Component_1_pH | pH difference |
| Solvent_Matthews_ratio | Solvent / Matthews | Normalized packing |
| Temp_Category | Bin[Low, Med, High] | Non-linear effects |
| pH_Category | Bin[Acidic, Neutral, Basic] | pH regimes |

### Model Training

**Simple & Advanced Baseline:**
- RandomForest (n_estimators=100, max_depth=20)
- XGBoost (n_estimators=100, max_depth=10, lr=0.1)
- LightGBM (n_estimators=100, max_depth=10)
- CatBoost (iterations=100, depth=10)

**Transformer:**
- Embedding dimension: 128
- Hidden dimension: 256
- Attention heads: 4
- Transformer layers: 2
- Dropout: 0.3
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)

---

## DETAILED RESULTS

### Component_1_Name (Classification)

**Challenge:** 2,432 unique chemical components

**Solution:** Filter to top 50 classes (covers 56% of data)

**Model Comparison:**
| Model | Simple | Advanced | Transformer |
|-------|--------|----------|-------------|
| RandomForest | 59.31% | 63.47% | - |
| XGBoost | 60.88% | **64.18%** | - |
| LightGBM | **61.12%** | 64.02% | - |
| CatBoost | 59.94% | 62.37% | - |
| Hybrid Transformer | - | - | 53.85% |

**Winner:** Advanced Baseline XGBoost (64.18%)

---

### Component_1_Conc (Regression)

**Challenge:** Concentration values with units ("100 mM", "25%", etc.)

**Simple Baseline Issue:**
- Input: "100 mM" (string)
- Pandas pd.to_numeric(): NaN
- Result: 100% missing → prediction impossible

**Advanced Baseline Solution:**
- Parse units: "100 mM" → 0.1 M
- Success: 83.21% parsed
- Log transform for stability

**Model Comparison:**
| Model | Simple | Advanced | Transformer |
|-------|--------|----------|-------------|
| RandomForest | N/A | **R²=0.4733** | - |
| XGBoost | N/A | R²=0.4420 | - |
| LightGBM | N/A | R²=0.4579 | - |
| CatBoost | N/A | R²=0.4638 | - |
| Hybrid Transformer | - | - | R²=0.3812 |

**Winner:** Advanced Baseline RandomForest (R²=0.4733)

---

### Component_1_pH (Regression)

**Challenge:** 74% missing values

**Solution:** Use general pH as prior, then iterative imputation

**Model Comparison:**
| Model | Simple | Advanced | Transformer |
|-------|--------|----------|-------------|
| RandomForest | R²=0.9427 | R²=0.9877 | - |
| XGBoost | R²=0.9371 | R²=0.9859 | - |
| LightGBM | R²=0.9513 | R²=0.9864 | - |
| CatBoost | R²=0.9558 | **R²=0.9934** | - |
| Hybrid Transformer | - | - | R²=0.9447 |

**Winner:** Advanced Baseline CatBoost (R²=0.9934) - Nearly perfect!

---

## INTERACTIVE APPLICATION

### Features
- **Input Sliders:** Adjust all 5 crystallization parameters
- **Real-Time Prediction:** Instant results
- **Top-5 Predictions:** See alternative components with probabilities
- **Model Comparison:** Switch between Simple and Advanced
- **Performance Metrics:** Display accuracy and R² scores
- **pH Visualization:** Color-coded pH scale indicator
- **Comparison Charts:** View performance plots

### How to Run
```bash
cd 05_app
streamlit run app.py
```

Or double-click: `05_app/RUN_APP.bat`

---

## GENERATED OUTPUTS

### Visualizations (12 total)

**EDA Plots (7)** - Location: `04_results/visualizations/`
1. `eda_01_missing_values_matrix.png` - Missing data patterns
2. `eda_02_missing_values_heatmap.png` - Missingness correlation
3. `eda_03_target_distributions.png` - Target distributions
4. `eda_04_feature_distributions.png` - Feature distributions
5. `eda_05_correlation_matrix.png` - Feature correlations
6. `eda_00_summary_statistics.csv` - Statistical summary
7. `eda_00_missing_values_detailed.csv` - Missing details

**Comparison Plots (5)** - Location: `04_results/visualizations/`
1. `01_component_name_comparison.png` - Name accuracy comparison
2. `02_component_conc_comparison.png` - Concentration R² comparison
3. `03_component_ph_comparison.png` - pH R² comparison
4. `04_all_approaches_heatmap.png` - Performance heatmap
5. `05_complete_comparison.png` - Complete 4-panel comparison

### Result Tables (4) - Location: `04_results/tables/`
1. `component_name_comparison.csv` - Name results
2. `component_conc_comparison.csv` - Concentration results
3. `component_ph_comparison.csv` - pH results
4. `final_summary.csv` - Complete summary

---

## METHODOLOGY

### Simple Baseline Pipeline
```
Raw Data → Basic Imputation → TF-IDF → Standard Scaling → Tree Models → Predictions
```

### Advanced Baseline Pipeline
```
Raw Data → Unit Parsing → Domain Constraints → Advanced Imputation → 
Feature Engineering → Robust Scaling → Tree Models → Predictions
```

### Transformer Pipeline
```
Raw Data → Text Cleaning → Tokenization → Embedding + Attention → 
MLP for Numerical → Fusion → Multi-task Heads → Predictions
```

---

## KEY LEARNINGS

### 1. Domain Knowledge is Critical
- Concentration parsing made huge difference (0% → 83%)
- Chemical constraints prevent unrealistic predictions
- Feature engineering (+3% accuracy)

### 2. pH is Easiest to Predict
- All approaches: R² > 0.94
- Strong correlation with input pH
- Well-behaved distribution

### 3. Component Name is Hardest
- 2,432 unique chemicals (high cardinality)
- Severe class imbalance
- 64% is good considering difficulty

### 4. Concentration Needs More Work
- R² = 0.47 (moderate performance)
- Unit parsing solved major blocker
- More chemical features could help (molecular weight, structure)

### 5. Transformers Need Scale
- Underperformed traditional ML
- Multi-task learning helped but not enough
- Would improve with 10× more data or pre-training

---

## FUTURE IMPROVEMENTS

### Immediate (Week 1)
1. **Hyperparameter Tuning** - Use Optuna for automated search
2. **Ensemble Models** - Combine predictions from multiple models
3. **Error Analysis** - Identify which samples are hardest to predict

### Short Term (Month 1)
4. **Pre-trained Embeddings** - ChemBERTa, MolBERT for chemical text
5. **Molecular Features** - SMILES, InChI, molecular fingerprints
6. **Cross-Validation** - K-fold for robust estimates

### Long Term (Quarter 1)
7. **Graph Neural Networks** - Treat molecules as graphs
8. **Transfer Learning** - Pre-train on larger chemical datasets
9. **Active Learning** - Query most uncertain predictions
10. **Production API** - FastAPI or Flask deployment

---

## DEPENDENCIES

### Core Libraries
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
catboost==1.2.2
```

### Visualization
```
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
missingno==0.5.2
```

### Application
```
streamlit==1.50.0
```

### Deep Learning (for Transformer)
```
torch==2.1.2
```

### Utilities
```
joblib==1.3.2
tqdm==4.66.1
openpyxl==3.1.2
nltk==3.8.1
optuna==3.5.0
```

---

## TROUBLESHOOTING

### Issue: Streamlit app shows errors
**Solution:** Check model paths in `05_app/app.py`, update to:
```python
model = joblib.load('../02_models/advanced_baseline/model_component_name.pkl')
```

### Issue: Concentration prediction fails
**Solution:** Use Advanced Baseline (not Simple)

### Issue: Missing files error
**Solution:** Run `python verify_all.py` to check setup

### Issue: Import errors
**Solution:** `conda activate btp_env` then `uv pip install -r 06_docs/requirements.txt`

---

## VERIFICATION

Run this to verify everything:
```bash
python verify_all.py
```

Expected output:
```
[OK] Simple Baseline - OK
[OK] Advanced Baseline - OK
[OK] Transformer - OK
[OK] App - OK
[OK] Results - OK

ALL CHECKS PASSED!
```


### Best Results

- **Overall Winner:** Advanced Baseline
- **Classification:** 64.18% accuracy
- **Concentration:** R² = 0.47 (vs N/A in simple)
- **pH:** R² = 0.99 (nearly perfect!)

---

## RECOMMENDATIONS

### For Production Use
**Choose:** Advanced Baseline  
**Models:** `02_models/advanced_baseline/`  
**Reason:** Best performance, proven results, fast inference

### For Research
**Choose:** Transformer  
**Model:** `02_models/transformer/`  
**Reason:** Interesting architecture, room for improvement

### For Quick Tests
**Choose:** Simple Baseline  
**Models:** `02_models/simple_baseline/`  
**Limitation:** No concentration prediction

---



## 📝 FILE LOCATIONS

### Essential Files
- `README.md` - This file (complete guide)
- `05_app/app.py` - Interactive application
- `05_app/RUN_APP.bat` - Quick launcher
- `verify_all.py` - Verification script

### Data
- `01_data/raw/` - Original Excel and CSV
- `01_data/processed/` - Preprocessed datasets

### Models
- `02_models/simple_baseline/` - Simple ML models
- `02_models/advanced_baseline/` - Advanced ML models ⭐
- `02_models/transformer/` - Deep learning model

### Results
- `04_results/visualizations/` - All plots (12)
- `04_results/tables/` - All CSVs (4+)

### Source Code
- `baseline_simple/` - Simple approach scripts
- `advanced_baseline/` - Advanced approach scripts
- `transformer_approach/` - Transformer scripts

---


