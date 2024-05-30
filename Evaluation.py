# evaluation.py

"""
Evaluation Module for Hawkes Processes for Event Prediction

This module contains functions for evaluating the performance of event prediction models
using appropriate metrics.

Metrics Used:
- Precision
- Recall
- F1-score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Libraries/Tools:
- scikit-learn
- pandas
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

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
        Prepare the data for evaluation by splitting into features and target.
        
        :param data: DataFrame, input data
        :param feature_columns: list, column names for features
        :param target_column: str, column name for target variable
        :return: DataFrame, Series, features and target for evaluation
        """
        X = data[feature_columns]
        y = data[target_column]
        return X, y

    def evaluate_model(self, X, y, model_name):
        """
        Evaluate an event prediction model using precision, recall, F1-score, MAE, and RMSE.
        
        :param X: DataFrame, feature matrix
        :param y: Series, true labels
        :param model_name: str, name of the model to evaluate
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        y_pred = model.predict(X)

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

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
    model_names = ['logistic_regression', 'random_forest']  # Models to evaluate

    evaluator = ModelEvaluation()

    # Load preprocessed data
    data = evaluator.load_data(processed_data_filepath)

    # Prepare data for evaluation
    X, y = evaluator.prepare_data(data, feature_columns, target_column)

    # Evaluate models
    for model_name in model_names:
        evaluator.evaluate_model(X, y, model_name)
    print("Model evaluation completed.")
