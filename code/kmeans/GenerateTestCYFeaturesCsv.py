from pathlib import Path
import pandas as pd
import numpy as np


# Settings.
num_rows = 200
output_file = Path("../../data/synthetic/test_cyfeatures.csv")


def generate_random_dataframe(n_rows, column_params):
    """
    Create a pandas DataFrame with N columns, each filled with random floats 
    from a normal distribution with given mean (mu) and standard deviation (sigma).
    Then take the absolute value of all values in the DataFrame.

    Parameters:
    n_rows (int): Number of rows in the DataFrame.
    column_params (dict): Dictionary where keys are column names and values are 
                          (mu, sigma) tuples defining the normal distribution.

    Returns:
    pd.DataFrame: Generated DataFrame.
    """
    data = {
        col_name: np.abs(np.random.normal(mu, sigma, n_rows))
        for col_name, (mu, sigma) in column_params.items()
    }

    return pd.DataFrame(data)


def generate_custom_cyears(n_rows, start_year=2025, num_cm=6, num_em=4):
    """
    Generate a list of strings formatted as 'YYYY_C_E', where:
    - E (economic model) cycles from 1 to num_em
    - C (climate model) cycles from 1 to num_cm
    - YYYY (year) starts at `start_year` and increases as needed

    Parameters:
    n_rows (int): Number of rows to generate
    start_year (int): The starting year (default: 2025)

    Returns:
    list: List of formatted date-like strings.
    """
    dates = []
    year, cm, em = start_year, 1, 1

    for _ in range(n_rows):
        dates.append(f"{year}_{cm}_{em}")
        em += 1
        if em > num_em:
            em = 1
            cm += 1
        if cm > num_cm:
            cm = 1
            year += 1

    return dates


column_settings1 = {
    'F1': (0., 1.),
    'F2': (5., 2.),
    'F3': (-3., 0.5),
    'F4': (0., 1.),
    'F5': (5., 2.),
    'F6': (30., 0.5),
    'F7': (0., 1.),
    'F8': (5., 2.),
    'F9': (200., 10.),
    'F10': (100., 2.),
    'F11': (5., 2.),
    'F12': (3., 0.5),
    'F13': (0.2, 1.8),
    'F14': (543, 2.),
    'F15': (344, 11.),
    'F16': (32., 43.),
}

column_settings = {
    'F1': (0., 10.),
    'F2': (5., 11.3),
    'F3': (-3., 1.1),
    'F4': (0., 0.9),
    'F5': (5., 1.2),
    'F6': (30., 0.6),
    'F7': (0., 1.),
    'F8': (5., 1.1),
}

df = generate_random_dataframe(num_rows, column_settings)
cyear_column = generate_custom_cyears(num_rows)
df.insert(0, "cy_id", cyear_column)


print(df.tail(20))


df.to_csv(output_file, index=False)
