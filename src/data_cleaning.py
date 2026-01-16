"""
data_cleaning.py

This module handles all data cleaning operations required before feature engineering and model training. 
The goal is to keep data prepration logic seperate from notebooks and make the pipeline resuable and production ready

"""
import pandas as pd

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning on the raw dataset. 

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
#  Cleaning logic will be added step by step

    return df
