"""Data preparation helpers for California housing dataset.

Provides `add_high_value_flag` to create a binary `HighValue` column
based on the median of `MedHouseValue`.
"""
from typing import Tuple
from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_housing_data() -> pd.DataFrame:
    """Load California housing dataset into a DataFrame."""
    housing_bunch = fetch_california_housing(as_frame=True)
    df = pd.DataFrame(housing_bunch.frame)
    return df


def add_high_value_flag(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Add a 'HighValue' column to the DataFrame based on MedHouseValue median.
    If median_value is not provided, it is computed from the DataFrame.

    Args:
        df: Input DataFrame containing 'MedHouseVal' column.
    Returns:
        df: Output DataFrame containing HighValue column
        value_counts: value counts for column HighValue

    """
    median_value = df['MedHouseVal'].median()
    df['HighValue'] = (df['MedHouseVal'] > median_value).astype(int)

    return df, df['HighValue'].value_counts()
