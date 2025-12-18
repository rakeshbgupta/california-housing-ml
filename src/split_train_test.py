from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the DataFrame into training and testing sets.

    Args:
        X: Input Feature set to be split.
        y: Output Label
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split
        startify: whether to stratify by y
    Returns:
        X_train, X_test, y_train, y_test    
        """
    stratify = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return X_train, X_test, y_train, y_test
