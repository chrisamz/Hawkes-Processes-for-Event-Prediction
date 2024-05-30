# hawkes_processes.py

"""
Hawkes Processes Module for Event Prediction

This module contains functions for developing and fitting Hawkes processes to capture
the temporal dependencies between events.

Techniques Used:
- Parameter estimation
- Self-exciting processes

Libraries/Tools:
- tick
"""

import pandas as pd
import numpy as np
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern
from tick.hawkes.inference import HawkesExpKern, HawkesSumExpKern
import joblib
import matplotlib.pyplot as plt

class HawkesProcesses:
    def __init__(self, decay=1.0):
        """
        Initialize the HawkesProcesses class.
        
        :param decay: float, decay parameter for the Hawkes process
        """
        self.decay = decay
        self.model = None

    def load_data(self, filepath):
        """
        Load preprocessed event data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def prepare_data(self, data, time_column):
        """
        Prepare the data for Hawkes process modeling by converting to event sequences.
        
        :param data: DataFrame, input data
        :param time_column: str, column name for timestamps
        :return: list, list of event sequences for each dimension
        """
        data[time_column] = pd.to_datetime(data[time_column])
        data.sort_values(by=time_column, inplace=True)
        start_time = data[time_column].min()
        data['time_since_start'] = (data[time_column] - start_time).dt.total_seconds()

        # Convert to list of event sequences for each dimension
        event_sequences = []
        for col in data.columns:
            if col not in [time_column, 'time_since_start']:
                event_times = data.loc[data[col] == 1, 'time_since_start'].values
                event_sequences.append(event_times.tolist())
        return event_sequences

    def fit_model(self, event_sequences):
        """
        Fit a Hawkes process model to the event sequences.
        
        :param event_sequences: list, list of event sequences for each dimension
        """
        self.model = HawkesExpKern(decay=self.decay)
        self.model.fit(event_sequences)
        print("Hawkes process model fitted.")

    def plot_intensity(self, event_sequences, end_time):
        """
        Plot the intensity function of the fitted Hawkes process.
        
        :param event_sequences: list, list of event sequences for each dimension
        :param end_time: float, end time for the simulation
        """
        hawkes_sim = SimuHawkesExpKernels(self.decay, end_time=end_time, verbose=False)
        hawkes_sim.track_intensity(0.1)
        hawkes_sim.simulate()
        fig, ax = plt.subplots(self.model.n_nodes, 1, figsize=(10, 8), sharex=True)
        for i in range(self.model.n_nodes):
            ax[i].plot(hawkes_sim.tracked_intensity_times, hawkes_sim.tracked_intensity[i], label=f'Intensity {i}')
            ax[i].legend()
        plt.show()

    def save_model(self, filepath):
        """
        Save the fitted Hawkes process model to a file.
        
        :param filepath: str, path to the file
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a fitted Hawkes process model from a file.
        
        :param filepath: str, path to the file
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    processed_data_filepath = 'data/processed/preprocessed_events.csv'
    model_filepath = 'models/hawkes_model.pkl'
    time_column = 'timestamp'

    hp = HawkesProcesses(decay=1.0)

    # Load preprocessed data
    data = hp.load_data(processed_data_filepath)

    # Prepare data for Hawkes process modeling
    event_sequences = hp.prepare_data(data, time_column)

    # Fit the Hawkes process model
    hp.fit_model(event_sequences)

    # Plot the intensity function
    hp.plot_intensity(event_sequences, end_time=3600)  # Example end time: 3600 seconds

    # Save the model
    hp.save_model(model_filepath)
    print("Hawkes process modeling completed and model saved.")
