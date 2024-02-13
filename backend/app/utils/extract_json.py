import pandas as pd
import json
import numpy as np

df = pd.read_csv('C:/Users/HP/Desktop/EstateVerse-Base-Model/backend/data/preprocessed_data.csv')

def convert_to_python_bool(value):
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def df_to_json(df):
    # Convert boolean values to standard Python booleans
    df = df.applymap(convert_to_python_bool)

    # Extract column headers
    columns = df.columns.tolist()

    # Take the first row of the DataFrame
    first_row = df.iloc[0].tolist()

    # Create a dictionary using column headers as keys and values from the first row
    data_dict = dict(zip(columns, first_row))

    # Convert the dictionary to JSON
    json_data = json.dumps(data_dict, indent=2, default=str)  # Use default=str to handle non-serializable types

    return json_data

json_data = df_to_json(df)

# Print the generated JSON
print(json_data)
