# event_prediction.py

"""
Event Prediction Module for Hawkes Processes for Event Prediction

This module contains functions for building and training prediction models to forecast future events
based on Hawkes processes and Granger causal inference analysis.

Techniques Used:
- Predictive modeling
- Time series forecasting

Libraries/Tools:
- scikit-learn
- tick
- pandas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import joblib

class EventPrediction:
    def __init__(self):
        """
        Initialize the EventPrediction class.
        """
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }

    def load_data(self, filepath):
        """
        Load preprocessed event data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def prepare_data(self, data, feature_columns, target_column):
        """
        Prepare the data for event prediction by splitting into features and target.
        
        :param data: DataFrame, input data
        :param feature_columns: list, column names for features
        :param target_column: str, column name for target variable
        :return: DataFrame, DataFrame, Series, Series, training and testing sets
        """
        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_name):
        """
        Train an event prediction model.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param model_name: str, name of the model to train
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} is not defined.")
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{model_name}_model.pkl')
        print(f"{model_name} model trained and saved.")

    def evaluate_model(self, X_test, y_test, model_name):
        """
        Evaluate an event prediction model using precision, recall, F1-score, MAE, and RMSE.
        
        :param X_test: DataFrame, testing feature matrix
        :param y_test: Series, true labels for testing
        :param model_name: str, name of the model to evaluate
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mae': mae,
            'rmse': rmse
        }
        print(f"{model_name} model evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    processed_data_filepath = 'data/processed/preprocessed_events.csv'
    feature_columns = ['feature_1', 'feature_2', 'feature_3']  # Example feature columns
    target_column = 'event_occurred'  # Example target column

    ep = EventPrediction()

    # Load preprocessed data
    data = ep.load_data(processed_data_filepath)

    # Prepare data for event prediction
    X_train, X_test, y_train, y_test = ep.prepare_data(data, feature_columns, target_column)

    # Train and evaluate logistic regression model
    ep.train_model(X_train, y_train, 'logistic_regression')
    ep.evaluate_model(X_test, y_test, 'logistic_regression')

    # Train and evaluate random forest model
    ep.train_model(X_train, y_train, 'random_forest')
    ep.evaluate_model(X_test, y_test, 'random_forest')

    print("Event prediction modeling completed.")
