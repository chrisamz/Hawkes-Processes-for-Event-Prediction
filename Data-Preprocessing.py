# data_preprocessing.py

"""
Data Preprocessing Module for Hawkes Processes for Event Prediction

This module contains functions for collecting, cleaning, normalizing, and preparing
event data for further analysis and modeling.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- datetime
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        pass

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data

    def normalize_data(self, data, columns):
        """
        Normalize the specified columns in the data.
        
        :param data: DataFrame, input data
        :param columns: list, columns to be normalized
        :return: DataFrame, normalized data
        """
        for column in columns:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        return data

    def extract_features(self, data, time_column, event_columns):
        """
        Extract relevant features from the data, such as timestamps and event types.
        
        :param data: DataFrame, input data
        :param time_column: str, column name for timestamps
        :param event_columns: list, column names for event types
        :return: DataFrame, data with extracted features
        """
        data[time_column] = pd.to_datetime(data[time_column])
        data['hour'] = data[time_column].dt.hour
        data['day'] = data[time_column].dt.day
        data['month'] = data[time_column].dt.month
        data['year'] = data[time_column].dt.year
        data = pd.get_dummies(data, columns=event_columns)
        return data

    def preprocess(self, filepath, time_column, event_columns, normalize_columns):
        """
        Execute the full preprocessing pipeline.
        
        :param filepath: str, path to the input data file
        :param time_column: str, column name for timestamps
        :param event_columns: list, column names for event types
        :param normalize_columns: list, column names to be normalized
        :return: DataFrame, preprocessed data
        """
        data = self.load_data(filepath)
        data = self.clean_data(data)
        data = self.extract_features(data, time_column, event_columns)
        data = self.normalize_data(data, normalize_columns)
        return data

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/events.csv'
    processed_data_filepath = 'data/processed/preprocessed_events.csv'
    time_column = 'timestamp'
    event_columns = ['event_type']
    normalize_columns = ['some_numeric_column']

    preprocessing = DataPreprocessing()

    # Preprocess the data
    preprocessed_data = preprocessing.preprocess(raw_data_filepath, time_column, event_columns, normalize_columns)
    preprocessed_data.to_csv(processed_data_filepath, index=False)
    print("Data preprocessing completed and saved to 'data/processed/preprocessed_events.csv'.")
