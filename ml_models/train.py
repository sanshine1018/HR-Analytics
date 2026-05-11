"""
Machine Learning Model Training
Train 3 models: Logistic Regression, Random Forest, Gradient Boosting
Select best model based on F1-score
Output: ml_models/models/best_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import json
from pathlib import Path

def load_engineered_data():
    """Load engineered data"""
    try:
        df = pd.read_csv("data/processed/engineered_data.csv")
        print(f"📊 Loaded engineered data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print("❌ Engineered data not found. Run feature_engineering.py first!")
        return None

def prepare_data(df):
    """Prepare data for model training"""
    print("\n🔄 Preparing data...")
    
    # Separate features and target
    # Look for common target columns (attrition, target, churn, left, etc.)
    target_cols = ['Attrition', 'attrition', 'target', 'Target', 'Churn', 'churn', 'Left', 'left']
    target_col = None
    
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # If no target found, use last column as target
        target_col = df.columns[-1]
        print(f"   ⚠️  No standard target column found. Using '{target_col}' as target")
    
    print(f"   ✓ Using '{target_col}' as target variable")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert target to numeric if needed
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    
    print(f"   ✓ Features: {X.shape[1]}")
    print(f"   ✓ Target classes: {len(np.unique(y))}")
    
    return X, y, target_col

def train_models(X, y):
    """Train multiple models"""
    print("\n🤖 Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n   Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
    
    return results, X_test, y_test

def select_best_model(results):
    """Select best model based on F1-score"""
    print("\n🏆 Selecting best model...")
    
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model_data = results[best_model_name]
    
    print(f"   ✓ Best Model: {best_model_name}")
    print(f"   ✓ F1-Score: {best_model_data['f1_score']:.4f}")
    
    return best_model_name, best_model_data['model']

def save_results(results, best_model_name, X_test, y_test):
    """Save model results and best model"""
    print("\n💾 Saving results...")
    
    # Save best model
    best_model = results[best_model_name]['model']
    model_path = "ml_models/models/best_model.pkl"
    pickle.dump(best_model, open(model_path, 'wb'))
    print(f"   ✓ Saved model: {model_path}")
    
    # Save results as JSON
    results_dict = {}
    for model_name, data in results.items():
        results_dict[model_name] = {
            'accuracy': float(data['accuracy']),
            'precision': float(data['precision']),
            'recall': float(data['recall']),
            'f1_score': float(data['f1_score'])
        }
    
    results_json_path = "data/outputs/model_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"   ✓ Saved results: {results_json_path}")
    
    # Save best model info
    best_model_info = {
        'best_model': best_model_name,
        'metrics': results_dict[best_model_name],
        'all_models': results_dict
    }
    
    best_model_path = "data/outputs/best_model_info.json"
    with open(best_model_path, 'w') as f:
        json.dump(best_model_info, f, indent=4)
    print(f"   ✓ Saved best model info: {best_model_path}")

def main():
    """Main execution"""
    print("=" * 50)
    print("   HR ANALYTICS - MODEL TRAINING")
    print("=" * 50)
    
    df = load_engineered_data()
    
    if df is not None:
        X, y, target_col = prepare_data(df)
        results, X_test, y_test = train_models(X, y)
        best_model_name, best_model = select_best_model(results)
        save_results(results, best_model_name, X_test, y_test)
        
        print("\n✨ Model training complete!")
        print(f"   Best model: {best_model_name}")

if __name__ == "__main__":
    main()
