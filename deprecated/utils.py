"""
Utility functions for working with pandas DataFrames and files.
"""

import pandas as pd
import os

def remove_prefix_from_columns(df, separator='_'):
    """
    Removes text before the specified separator character in all column names of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame whose column names will be modified
        separator (str, optional): Character that separates prefix from the rest of the column name.
                                  Default is '_'.

    Returns:
        pandas.DataFrame: A DataFrame with modified column names (original DataFrame is not modified)

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=['Q01_ID', 'Q02_Name', 'Q03_Age'])
        >>> remove_prefix_from_columns(df)
        # Returns DataFrame with columns ['ID', 'Name', 'Age']
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result = df.copy()

    # Create a dictionary mapping old column names to new column names
    new_columns = {}
    for col in result.columns:
        if separator in col:
            # Split the column name at the first occurrence of the separator
            # and take the part after the separator
            new_columns[col] = col.split(separator, 1)[1]
        else:
            # If the separator is not in the column name, keep it as is
            new_columns[col] = col

    # Rename the columns
    result.rename(columns=new_columns, inplace=True)

    return result

def verify_pre_post_files(filename):
    """
    Verifies that the filename contains '_PRE_' and that corresponding files
    with '_MID_' and '_POST_' instead of '_PRE_' exist.

    Args:
        filename (str): The filename to verify

    Returns:
        tuple: A tuple containing (is_valid, message)
            - is_valid (bool): True if the filename contains '_PRE_' and the corresponding '_MID_' and '_POST_' files exist
            - message (str): A message explaining the validation result

    Examples:
        >>> verify_pre_post_files('../ECLASS_Ita_PRE_FIS2a.csv')
        # If MID and POST files exist: (True, 'Valid PRE file with corresponding MID and POST files')
        # If files don't exist: (False, 'Corresponding MID or POST file not found')
    """
    # Check if filename contains '_PRE_'
    if '_PRE_' not in filename:
        return False, f"Filename '{filename}' does not contain '_PRE_'"

    # Generate the corresponding MID and POST filenames
    mid_filename = filename.replace('_PRE_', '_MID_')
    post_filename = filename.replace('_PRE_', '_POST_')

    # Check if the MID file exists
    if not os.path.exists(mid_filename):
        return False, f"Corresponding MID file '{mid_filename}' not found"

    # Check if the POST file exists
    if not os.path.exists(post_filename):
        return False, f"Corresponding POST file '{post_filename}' not found"

    return True, f"Valid PRE file with corresponding MID file '{mid_filename}' and POST file '{post_filename}'"

# Example usage
if __name__ == "__main__":
    # Example for remove_prefix_from_columns
    df = pd.DataFrame(columns=['Q01_ID', 'Q02_Name', 'Q03_Age', 'Other_Column'])
    df_new = remove_prefix_from_columns(df)
    print("Original columns:", df.columns.tolist())
    print("New columns:", df_new.columns.tolist())

    # Example for verify_pre_post_files
    test_file = '../ECLASS_Ita_PRE_FIS2a.csv'
    is_valid, message = verify_pre_post_files(test_file)
    print(f"\nVerifying file '{test_file}':")
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")
