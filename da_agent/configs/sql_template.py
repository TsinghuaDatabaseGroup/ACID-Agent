"""SQL Template Module - helper script template for SQL command execution.

This module defines the Python script template used by the SQL action:
- Opens SQLite database files inside the workspace
- Executes SQL queries through pandas
- Saves query results to CSV or prints them directly

References:
- https://github.com/yiyihum/da-code/tree/main/da_agent/configs/sql_template.py
"""

SQL_TEMPLATE = """
import sqlite3
import pandas as pd
import os

def execute_sql(file_path, command, output_path):
    # make sure the file path is correct
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {{file_path}}")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(file_path)
    
    try:
        # Execute the SQL command and fetch the results
        df = pd.read_sql_query(command, conn)
        
        # Check if the output should be saved to a CSV file or printed directly
        if output_path.lower().endswith(".csv"):
            df.to_csv(output_path, index=False)
            print(f"Output saved to: {{output_path}}")
        else:
            print(df)
    except Exception as e:
        print(f"ERROR: {{e}}")
    finally:
        # Close the connection to the database
        conn.close()

# Example usage
file_path = "{file_path}"  # Path to your SQLite database file
command = "{code}"             # SQL command to be executed
output_path = "{output}" # Path to save the output as a CSV or "directly"

execute_sql(file_path, command, output_path)

"""
