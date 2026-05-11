# HR Analytics Pipeline

A simple data pipeline for HR analytics with data processing, exploratory analysis, machine learning models, and Tableau integration.

## 📁 Project Structure

```
HR-Analytics/
├── data/
│   ├── raw/              # Place your HR data files here (.csv, .xlsx)
│   ├── processed/        # Cleaned data output
│   └── outputs/          # Final results for Tableau & predictions
├── scripts/
│   ├── data_cleaning.py      # Step 1: Clean and process raw data
│   ├── eda.py                # Exploratory Data Analysis
│   └── feature_engineering.py # Step 2: Feature creation & scaling
├── ml_models/
│   ├── train.py          # Step 3: Train machine learning models
│   ├── predict.py        # Step 4: Make predictions
│   └── models/           # Saved trained models
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Add Your Data**
Place your HR dataset in `data/raw/` folder (CSV or Excel format)

### 3. **Run Pipeline**
Execute steps in order:

```bash
# Step 1: Clean data
python scripts/data_cleaning.py

# Step 2: Exploratory Analysis
python scripts/eda.py

# Step 3: Feature Engineering
python scripts/feature_engineering.py

# Step 4: Train ML Models
python ml_models/train.py

# Step 5: Make Predictions
python ml_models/predict.py
```

## 📊 What Each Script Does

| Script | Purpose |
|--------|---------|
| `data_cleaning.py` | Load raw data, remove duplicates, handle missing values, fix data types |
| `eda.py` | Generate charts and statistics for insights |
| `feature_engineering.py` | Create new features, scale data for ML models |
| `train.py` | Train 3 ML models and save the best one |
| `predict.py` | Use trained model to make predictions |

## 📈 Tableau Integration

All processed data and predictions are saved in `data/outputs/`:
- `processed_data.csv` - Clean data for Tableau dashboards
- `predictions.csv` - Model predictions for analysis
- `model_results.json` - Model performance metrics

## 🤖 Machine Learning Models

The pipeline trains and compares:
1. **Logistic Regression** - Simple baseline
2. **Random Forest** - Best for HR patterns
3. **Gradient Boosting** - Advanced predictions

Best model is automatically selected and saved.

## 📝 Example Use Case

Perfect for:
- Employee attrition prediction
- Salary analysis
- Department performance insights
- Promotion probability
- Performance categorization

---

**Author:** sanshine1018  
**Last Updated:** 2026-05-11
