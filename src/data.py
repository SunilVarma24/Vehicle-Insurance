# src/data.py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the vehicle insurance preprocessed dataset."""
    df = pd.read_csv(filepath)
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from the Annual_Premium column."""
    q1 = np.quantile(df.Annual_Premium, 0.25)
    q3 = np.quantile(df.Annual_Premium, 0.75)
    iqr = q3 - q1

    outliers = df[(df.Annual_Premium < q1 - 1.5 * iqr) | (df.Annual_Premium > q3 + 1.5 * iqr)]
    df_new = df[(df.Annual_Premium > q1 - 1.5 * iqr) & (df.Annual_Premium < q3 + 1.5 * iqr)]
    
    # Display outlier counts by Response
    print(pd.DataFrame({
        "New Data": [len(df_new[df_new.Response == 0]), len(df_new[df_new.Response == 1])],
        "Outliers": [len(outliers[outliers.Response == 0]), len(outliers[outliers.Response == 1])]
    }))
    return df_new

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering including value replacements and column dropping."""
    df_new = df.copy()
    # Replace Gender values
    df_new.loc[df_new['Gender'] == 'Female', 'Gender'] = 0
    df_new.loc[df_new['Gender'] == 'Male', 'Gender'] = 1
    df_new['Gender'] = df_new['Gender'].astype(int)

    # Replace Vehicle_Age values
    df_new.loc[df_new['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
    df_new.loc[df_new['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
    df_new.loc[df_new['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
    df_new['Vehicle_Age'] = df_new['Vehicle_Age'].astype(int)

    # Replace Vehicle_Damage values
    df_new.loc[df_new['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
    df_new.loc[df_new['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
    df_new['Vehicle_Damage'] = df_new['Vehicle_Damage'].astype(int)

    # Drop the 'id' column as it is not needed
    df_new.drop('id', axis=1, inplace=True)
    
    return df_new