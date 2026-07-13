import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will...")

def main():
    print("Loading datasets")
    file_path = "train.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: cannot find '{file_path}'")
        return
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded sucessfully. Rows: {df.shape[0]}, Features: {df.shape[1]}")

    print("Handling missing ")
    df.loc[0:25, 'H'] = np.nan

    imputer = SimpleImputer(strategy='median')

    df['H'] = imputer.fit_transform(df[['H']])
    print(f"Imputation complete. 'Hits' (H) now has {df['H'].isnull()}")

    df["LogRuns"] = np.log1p(df["R"])
    print(f"Log Transformations applied. New skewness: {df['LogRuns'].skew():.2f} (closer to 0 is perfectly balanced)")

if __name__ == "__main__":
    main()