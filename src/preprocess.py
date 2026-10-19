import datetime
import pandas as pd

def perform_eda_summary(df: pd.DataFrame):
    """Performs and prints exploratory data analysis insights."""
    print("=" * 50)
    print("      EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 50)
    print(f"Total Rows & Columns: {df.shape}")
    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())
    print("\n--- Statistical Summary of Numerical Features ---")
    print(df.describe())
    print("=" * 50)

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data and extracts high-impact features utilizing every column 
    (Year, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner)[cite: 2].
    """
    data = raw_df.copy()
    
    # 1. Run EDA logging
    perform_eda_summary(data)
    
    # 2. Temporal feature: Convert Year into Car Age
    current_year = datetime.datetime.now().year
    data['Car_Age'] = current_year - data['Year']
    data = data.drop(columns=['Year'])
    
    # 3. Extract Brand from Car Name for categorization
    if 'Car_Name' in data.columns:
        data['Brand'] = data['Car_Name'].apply(lambda x: str(x).split()[0].lower())
        data = data.drop(columns=['Car_Name'])
        
    # 4. Advanced Interaction Features (Ensures Owner, Mileage, and Age impact predictions jointly)
    data['Kms_Per_Year'] = data['Driven_kms'] / (data['Car_Age'] + 1)
    data['Usage_Intensity'] = data['Driven_kms'] * data['Car_Age']
    
    # 5. Owner impact weighting factor
    data['Owner_Penalty_Factor'] = data['Owner'] * data['Present_Price']
    
    return data