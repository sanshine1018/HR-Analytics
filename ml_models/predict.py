"""
Machine Learning Predictions
Use trained best model to make predictions on full dataset
Output: data/outputs/predictions.csv (Ready for Tableau)
"""

import pandas as pd
import numpy as np
import pickle
import json

def load_data_and_model():
    """Load engineered data and trained model"""
    try:
        df = pd.read_csv("data/processed/engineered_data.csv")
        model = pickle.load(open("ml_models/models/best_model.pkl", 'rb'))
        print(f"📊 Loaded data: {len(df)} rows")
        print(f"🤖 Loaded trained model")
        return df, model
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure to run train.py first!")
        return None, None

def make_predictions(X, model):
    """Generate predictions"""
    print("\n🎯 Generating predictions...")
    
    # Get predictions
    predictions = model.predict(X)
    
    # Get prediction probabilities
    try:
        probabilities = model.predict_proba(X)
        print(f"   ✓ Got probabilities from model")
    except:
        # If model doesn't have predict_proba, create dummy probabilities
        probabilities = np.column_stack([1 - predictions, predictions])
        print(f"   ⚠️  Model doesn't support probability. Using 0/1 probabilities")
    
    print(f"   ✓ Predicted {len(predictions)} samples")
    
    return predictions, probabilities

def create_prediction_dataframe(df, predictions, probabilities):
    """Create dataframe with predictions and probabilities"""
    print("\n📊 Creating prediction dataframe...")
    
    # Determine number of classes
    n_classes = probabilities.shape[1]
    
    pred_df = df.copy()
    pred_df['Prediction'] = predictions
    
    # Add probability columns
    for i in range(n_classes):
        pred_df[f'Probability_Class_{i}'] = probabilities[:, i]
    
    # Create confidence level
    max_prob = np.max(probabilities, axis=1)
    pred_df['Confidence'] = max_prob
    
    # Create prediction level (High/Medium/Low)
    pred_df['Prediction_Level'] = pd.cut(pred_df['Confidence'], 
                                         bins=[0, 0.5, 0.75, 1.0], 
                                         labels=['Low', 'Medium', 'High'])
    
    print(f"   ✓ Created prediction dataframe with {len(pred_df.columns)} columns")
    print(f"   Columns: {', '.join(pred_df.columns[:10])}...")
    
    return pred_df

def save_predictions(pred_df):
    """Save predictions to CSV"""
    output_path = "data/outputs/predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"   Shape: {pred_df.shape}")
    print(f"   Ready for Tableau! 📊")
    
    return pred_df

def generate_summary_stats(pred_df):
    """Generate summary statistics"""
    print("\n📈 Prediction Summary:")
    
    # Prediction distribution
    pred_counts = pred_df['Prediction'].value_counts()
    print(f"   Prediction distribution:")
    for pred_val, count in pred_counts.items():
        pct = (count / len(pred_df)) * 100
        print(f"      Class {pred_val}: {count} samples ({pct:.1f}%)")
    
    # Confidence statistics
    print(f"\n   Confidence statistics:")
    print(f"      Mean: {pred_df['Confidence'].mean():.4f}")
    print(f"      Min:  {pred_df['Confidence'].min():.4f}")
    print(f"      Max:  {pred_df['Confidence'].max():.4f}")
    
    # Prediction level distribution
    print(f"\n   Prediction level distribution:")
    level_counts = pred_df['Prediction_Level'].value_counts()
    for level, count in level_counts.items():
        pct = (count / len(pred_df)) * 100
        print(f"      {level}: {count} samples ({pct:.1f}%)")
    
    # Save summary to JSON
    summary = {
        'total_predictions': len(pred_df),
        'prediction_distribution': pred_counts.to_dict(),
        'confidence_stats': {
            'mean': float(pred_df['Confidence'].mean()),
            'min': float(pred_df['Confidence'].min()),
            'max': float(pred_df['Confidence'].max()),
            'std': float(pred_df['Confidence'].std())
        },
        'prediction_levels': level_counts.to_dict()
    }
    
    summary_path = "data/outputs/prediction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n   ✓ Saved summary to: {summary_path}")

def main():
    """Main execution"""
    print("=" * 50)
    print("   HR ANALYTICS - PREDICTIONS FOR TABLEAU")
    print("=" * 50)
    
    df, model = load_data_and_model()
    
    if df is not None and model is not None:
        predictions, probabilities = make_predictions(df, model)
        pred_df = create_prediction_dataframe(df, predictions, probabilities)
        save_predictions(pred_df)
        generate_summary_stats(pred_df)
        
        print("\n✨ Predictions complete!")
        print("   📊 Ready to use in Tableau!")
        print(f"   📁 File: data/outputs/predictions.csv")
    else:
        print("\n❌ Could not generate predictions. Check errors above.")

if __name__ == "__main__":
    main()
