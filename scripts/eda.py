"""
Exploratory Data Analysis (EDA)
Generate charts and statistics from cleaned data
Output: Charts and statistics saved to data/outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_cleaned_data():
    """Load cleaned data"""
    try:
        df = pd.read_csv("data/processed/cleaned_data.csv")
        print(f"📊 Loaded cleaned data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print("❌ Cleaned data not found. Run data_cleaning.py first!")
        return None

def generate_statistics(df):
    """Generate basic statistics"""
    print("\n📈 Data Statistics:")
    print(f"   Shape: {df.shape}")
    print(f"   Data types:\n{df.dtypes}")
    print(f"\n   Descriptive stats:\n{df.describe()}")
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "missing_values": df.isnull().sum().to_dict()
    }
    
    return stats

def create_distributions(df):
    """Create distribution plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        print("\n📊 Creating distribution charts...")
        
        fig, axes = plt.subplots(nrows=(len(numeric_cols) + 1) // 2, ncols=2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df[col], bins=30, edgecolor='black', color='skyblue')
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        
        # Remove empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = "data/outputs/distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        print("📊 Creating correlation heatmap...")
        
        plt.figure(figsize=(12, 8))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        
        output_path = "data/outputs/correlation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()

def create_categorical_charts(df):
    """Create bar charts for categorical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print("📊 Creating categorical charts...")
        
        fig, axes = plt.subplots(nrows=(len(categorical_cols) + 1) // 2, ncols=2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            if len(df[col].unique()) <= 20:  # Only plot if reasonable number of categories
                df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Count")
                axes[idx].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for idx in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = "data/outputs/categorical_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()

def save_statistics_json(stats):
    """Save statistics as JSON"""
    output_path = "data/outputs/eda_statistics.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✓ Saved: {output_path}")

def main():
    """Main execution"""
    print("=" * 50)
    print("   HR ANALYTICS - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    df = load_cleaned_data()
    
    if df is not None:
        # Generate statistics
        stats = generate_statistics(df)
        save_statistics_json(stats)
        
        # Create visualizations
        create_distributions(df)
        create_correlation_heatmap(df)
        create_categorical_charts(df)
        
        print("\n✨ EDA complete! Check data/outputs/ for charts")

if __name__ == "__main__":
    main()
