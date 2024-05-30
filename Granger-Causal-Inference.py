# granger_causal_inference.py

"""
Granger Causal Inference Module for Event Prediction

This module contains functions for implementing Granger causal inference to identify
causal relationships between different event types.

Techniques Used:
- Time series analysis
- Hypothesis testing

Libraries/Tools:
- statsmodels
- pandas
"""

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

class GrangerCausalInference:
    def __init__(self, max_lag=10):
        """
        Initialize the GrangerCausalInference class.
        
        :param max_lag: int, maximum number of lags to test for Granger causality
        """
        self.max_lag = max_lag

    def load_data(self, filepath):
        """
        Load preprocessed event data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath, parse_dates=['timestamp'])
        return data

    def prepare_data(self, data, event_columns):
        """
        Prepare the data for Granger causal inference by creating time series for each event type.
        
        :param data: DataFrame, input data
        :param event_columns: list, column names for event types
        :return: DataFrame, prepared data with time series for each event type
        """
        data.set_index('timestamp', inplace=True)
        prepared_data = data[event_columns]
        return prepared_data

    def granger_causality_matrix(self, data):
        """
        Create a matrix of Granger causality results for all pairs of event types.
        
        :param data: DataFrame, input data with time series for each event type
        :return: DataFrame, matrix of p-values for Granger causality tests
        """
        variables = data.columns
        matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for col in matrix.columns:
            for row in matrix.index:
                test_result = grangercausalitytests(data[[row, col]], max_lag=self.max_lag, verbose=False)
                p_values = [round(test_result[i+1][0]['ssr_chi2test'][1], 4) for i in range(self.max_lag)]
                min_p_value = np.min(p_values)
                matrix.loc[row, col] = min_p_value
        return matrix

    def plot_causality_matrix(self, matrix):
        """
        Plot the Granger causality matrix as a heatmap.
        
        :param matrix: DataFrame, matrix of p-values for Granger causality tests
        """
        plt.figure(figsize=(10, 8))
        plt.title('Granger Causality Matrix')
        plt.imshow(matrix, cmap='viridis', aspect='auto', interpolation='none')
        plt.colorbar(label='p-value')
        plt.xticks(ticks=np.arange(len(matrix.columns)), labels=matrix.columns, rotation=90)
        plt.yticks(ticks=np.arange(len(matrix.index)), labels=matrix.index)
        plt.show()

if __name__ == "__main__":
    processed_data_filepath = 'data/processed/preprocessed_events.csv'
    event_columns = ['event_type_1', 'event_type_2', 'event_type_3']  # Example event types

    gci = GrangerCausalInference(max_lag=10)

    # Load preprocessed data
    data = gci.load_data(processed_data_filepath)

    # Prepare data for Granger causal inference
    prepared_data = gci.prepare_data(data, event_columns)

    # Compute Granger causality matrix
    causality_matrix = gci.granger_causality_matrix(prepared_data)
    print("Granger Causality Matrix:\n", causality_matrix)

    # Plot the causality matrix
    gci.plot_causality_matrix(causality_matrix)
    print("Granger causality analysis completed.")
