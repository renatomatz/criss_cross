import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetype


def drop_n_rows(df, n):
    """Drop first n rows from data.

    Args:
        df: DataFrame to be modified.
        n: number of rows to drop.
    """

    if (not isinstance(n, int)) or (n < 0):
        raise TypeError("n must be a positive integer")

    df.drop(df.index[:n], inplace=True)

    return df


def drop_cols(df, cols):
    """ Drop selected columns inplace

    Args:
        df: DataFrame to modify.
        cols: Columns to drop.
    """

    if not isinstance(cols, list):
        cols = [cols]

    df.drop(cols, axis=1, inplace=True)

    return df


def keep_cols(df, cols):
    """ Keep selected columns inplace.

    Args:
        df: DataFrame to modify.
        cols: Columns to keep.
    """

    if not isinstance(cols, list):
        cols = [cols]

    dcols = [col for col in df.columns if col not in cols]

    df.drop(dcols, axis=1, inplace=True)

    return df


def rename_cols(df, new_names):
    """Rename selected columns inplace

    Args:
        df: DataFrame to modify.
        new_names: Either a string (If df is a series or has one columns), 
            a list (whose length must be the same as the number col columns in
            df or a dictionary with {old_name; new_name} pairs.
    """

    if isinstance(new_names, str):
        new_names = [new_names]

    if isinstance(new_names, list):
        
        if len(new_names) != len(df.columns):
            raise ValueError(f"if new_names is a list, it must be of the same length as dataframe columns, \
                which in this case is {len(df.columns)}")

        df.columns = new_names

    elif isinstance(new_names, dict):

        df.rename(new_names, axis=1, inplace=True)

    return df


def set_date_index(df, cols="Date", is_iloc=False, date_format=None):
    """Set the selected datetime column as the dataframe's index.

    Args:
        df: DataFrame to be modified
        col_name: Column to become index.
        is_iloc: whether the "cols" parameter is an integer location to the 
            column.
        date_format: datetime format to convert column, in case the column
            is currently currently of type string.
    """

    if is_iloc:
        if not isinstance(cols, int):
            TypeError("col_name must be an integer column location if you set is_iloc equal to True")

        if isinstance(cols, slice):
            cols = df.columns[cols].tolist()
        else:
            cols = df.columns[cols]

    if not is_datetype(df[cols]):
        df[cols] = pd.to_datetime(df[cols], format=None)

    df.set_index(cols, drop=True, inplace=True) 
    df.index.name = "Date"
    df.sort_index(inplace=True)

    return df


def remove_punct(df, punct, cols=None) :
    """Remove a certain punctuation from selected columns.

    NOTE: This function had problems in the past where the very last elemnt of
    a column is set to np.nan. This implementation checks for that, but adds
    some complexity and slows down the function.

    Args:
        df: DataFrame to modify.
        punct: Single punctuation or list of punctuation to remove.
        cols: columns to modify.
    """

    if cols is None:
        cols = df.columns.tolist()
    elif not isinstance(cols, list):
        cols = [cols]

    if not isinstance(punct, list):
        punct = [punct]

    for col in cols:
        if df[col].dtypes == "object":
            for p in punct:
                # TODO: you absolue monster
                # Anyhow, the .str.replace method deletes the last value
                # of a series from time to time.
                new_col = df[col].str.replace(p, "")
                if pd.isna(new_col[-1]):
                    new_col[-1] = df[col][-1]

                df[col] = new_col

    return df


def delete_item(df, item, cols=None) :
    """Delete specific items from the dataframe

    Args:
        df: DataFrame to modify.
        item: Item to find and delete.
        cols: Columns to delete items.
    """

    if cols is None:
        cols = df.columns.tolist()
    elif not isinstance(cols, list):
        cols = [cols]

    elif not isinstance(item, list):
        item = [item]

    for col in cols:
        df[col] = df[col].replace(item, np.nan)

    return df


def set_type(df, dtype, cols=None, is_iloc=False):
    """Set a column to a specific data type.

    Args:
        df: DataFrame to modify.
        dtype: Data type to change columns into.
        cols: columns to modify.
        is_iloc: whether the "cols" parameter is an integer location to the 
            column.
    """

    if not isinstance(dtype, type):
        raise TypeError("dtypep argument must of of type type")

    if is_iloc:
        if not isinstance(cols, (int, slice)):
            raise TypeError("col_name must be an integer column location if you set is_iloc equal to True")

        if isinstance(cols, slice):
            cols = df.columns[cols].tolist()
        else:
            cols = df.columns[cols]


    if cols is None:
        cols = df.columns.tolist()
    elif not isinstance(cols, list):
        cols = [cols]

    df[cols] = df[cols].astype(dtype)

    return df
