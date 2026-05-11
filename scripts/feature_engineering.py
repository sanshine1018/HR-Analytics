"""
Feature Engineering
Create features, encode categories, scale numeric data
Output: data/processed/engineered_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

def load_cleaned_data():
    """Load cleaned data"""
    try:
        df = pd.read_csv("data/processed/cleaned_data.csv")
        print(f"📊 Loaded data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print("❌ Cleaned data not found. Run data_cleaning.py first!")
        return None

def create_features(df):
    """Create new features from existing data"""
    print("\n⚙️  Creating features...")
    
    # Numeric features - create bins/quartiles for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Create quartile bins
        df[f'{col}_quartile'] = pd.qcut(df[col], q=4, duplicates='drop', labels=False)
    
    # Interaction features (example: multiply related numeric columns)
    if len(numeric_cols) >= 2:
        df['numeric_interaction'] = df[numeric_cols[0]] * df[numeric_cols[1]]
    
    print(f"   ✓ Created {len([c for c in df.columns if 'quartile' in c])} quartile features")
    print(f"   ✓ Total features now: {len(df.columns)}")
    
    return df

def encode_categorical(df):
    """Encode categorical variables"""
    print("\n🔤 Encoding categorical features...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        
        # Use label encoding for small number of categories
        if unique_vals <= 10:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            print(f"   ✓ Label encoded {col} ({unique_vals} categories)")
        else:
            # Use frequency encoding for many categories
            freq_encoding = df[col].value_counts(normalize=True).to_dict()
            df[f'{col}_encoded'] = df[col].map(freq_encoding)
            print(f"   ✓ Frequency encoded {col} ({unique_vals} categories)")
    
    return df

def scale_numeric_features(df):
    """Scale numeric features"""
    print("\n📊 Scaling numeric features...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    # Create new column names for scaled features
    for idx, col in enumerate(numeric_cols):
        df[f'{col}_scaled'] = scaled_data[:, idx]
    
    print(f"   ✓ Scaled {len(numeric_cols)} numeric columns")
    
    # Save scaler for later use
    pickle.dump(scaler, open('ml_models/models/scaler.pkl', 'wb'))
    
    return df

def remove_original_columns(df):
    """Remove original non-encoded categorical and non-scaled numeric columns"""
    print("\n🧹 Cleaning up original columns...")
    
    # Keep only encoded/scaled versions
    cols_to_drop = []
    
    # Drop original categorical columns (keep encoded versions)
    categorical_cols = df.select_dtypes(include=['object']).columns
    cols_to_drop.extend(categorical_cols)
    
    # Drop original numeric columns (keep scaled versions)
    # But keep the quartile versions
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if '_scaled' not in c and '_quartile' not in c and '_interaction' not in c and '_encoded' not in c]
    cols_to_drop.extend(numeric_cols)
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"   ✓ Dropped {len(cols_to_drop)} original columns")
    print(f"   ✓ Final features: {len(df.columns)}")
    
    return df

def save_engineered_data(df):
    """Save engineered data"""
    output_path = "data/processed/engineered_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Engineered data saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    return df

def main():
    """Main execution"""
    print("=" * 50)
    print("   HR ANALYTICS - FEATURE ENGINEERING")
    print("=" * 50)
    
    df = load_cleaned_data()
    
    if df is not None:
        df = create_features(df)
        df = encode_categorical(df)
        df = scale_numeric_features(df)
        df = remove_original_columns(df)
        save_engineered_data(df)
        
        print("\n✨ Feature engineering complete!")
        print("   Ready for machine learning models!")

if __name__ == "__main__":
    main()
