"""
data_cleaning.py

This module handles all data cleaning operations required before feature engineering and model training. 
The goal is to keep data prepration logic seperate from notebooks and make the pipeline resuable and production ready

"""

"""
    Steps Planned in this function:
    - Handle missing values 
    - Standardize column formats 
    - Remove obvious inconsistensies 
    - Prepare data for feature engineering
    
    Parameters: 
    df(pd.DataFrame): Raw input dataframe

    returns:
    pd.DataFrame : Cleaned DataFrame
   
    """

import pandas as pd

def clean_data(df):
    """
    Performs initial data cleaning on the raw dataset. 
    """
    # 1. Remove the duplicate rows to avoid bias in the model training 
    df=df.drop_duplicates()
    '''
    #1.1Check for the missing values 
    missing=df.isnull().sum()

    Note: While making, this line was used for inspection.
    It is not used in production. Here kept in comments for study purpose. 
    '''

    # 2. Standardize column names
    df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

    
    

    # 3. Drop rows where target variable is missing 
    if "price_euro" in df.columns:
        df = df.dropna(subset=["price_euro"])
   

    #4.1)---- RAM Cleaning----

    if "ram_gb" in df.columns:
        df["ram_gb"]=(
            df["ram_gb"]
            .astype(str)
            .str.replace("gb","",case=False, regex=False)
            .str.strip()
        )

    # 4.2)---- Weight cleaning: "2.3kg" -> 2.3 ----

    if "weight_kg" in df.columns:
        df["weight_kg"]=(
            df["weight_kg"]
            .astype(str)
            .str.replace("kg", "", case=False, regex=False)
            .str.strip()
        )
    # 4.3) ---- Inches cleaning: ensure numeric screen size ----

    if "inches" in df.columns:
        df["inches"]=(
            df["inches"]
            .astype(str)
            .str.strip()
        )

    #5) ScreenResolution Parsing

    if "screenresolution" in df.columns:
        resolution=df["screenresolution"].astype(str)
        df["screen_width"] = resolution.str.extract(r'(\d+)\s*x\s*(\d+)')[0]
        df["screen_height"] = resolution.str.extract(r'(\d+)\s*x\s*(\d+)')[1]
    

        
    #Ensure that numerical columns are numeric
    numeric_cols = [
    "inches",
    "ram_gb",
    "weight_kg",
    "price_euro",
    "screen_width",
    "screen_height"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col], errors="coerce")

    return df


