"""
Data Cleaning Script
Load raw HR data, handle missing values, remove duplicates, and fix data types
Output: data/processed/cleaned_data.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

def load_raw_data():
    """Load raw data from data/raw folder"""
    raw_folder = "data/raw"
    
    # Find CSV or Excel file
    csv_files = list(Path(raw_folder).glob("*.csv"))
    excel_files = list(Path(raw_folder).glob("*.xlsx")) + list(Path(raw_folder).glob("*.xls"))
    
    if csv_files:
        file_path = csv_files[0]
        print(f"📥 Loading CSV: {file_path}")
        df = pd.read_csv(file_path)
    elif excel_files:
        file_path = excel_files[0]
        print(f"📥 Loading Excel: {file_path}")
        df = pd.read_excel(file_path)
    else:
        print("❌ No CSV or Excel file found in data/raw/")
        print("   Please add your HR data file to: data/raw/")
        return None
    
    return df

def clean_data(df):
    """Clean the dataframe"""
    print("\n🧹 Cleaning data...")
    
    # Display initial info
    print(f"   Original rows: {len(df)}")
    print(f"   Original columns: {len(df.columns)}")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"   ✓ Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"   ✓ Found missing values in: {', '.join(missing_cols)}")
        
        # Fill numeric columns with median
        numeric_cols = df[missing_cols].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df[missing_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Remove outliers (numerical columns > 3 std)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[~((df[col] - mean).abs() > 3 * std)]
    
    print(f"   ✓ Cleaned rows: {len(df)}")
    
    return df

def save_cleaned_data(df):
    """Save cleaned data"""
    output_path = "data/processed/cleaned_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")
    return df

def main():
    """Main execution"""
    print("=" * 50)
    print("   HR ANALYTICS - DATA CLEANING")
    print("=" * 50)
    
    create_directories()
    df = load_raw_data()
    
    if df is not None:
        df = clean_data(df)
        save_cleaned_data(df)
        print("\n✨ Data cleaning complete!")
    else:
        print("\n⚠️  Skipping cleaning. Add your HR data file to data/raw/")

if __name__ == "__main__":
    main()
