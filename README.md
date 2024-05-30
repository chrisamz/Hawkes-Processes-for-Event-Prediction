# Hawkes Processes for Event Prediction

## Description

This project focuses on developing models to predict the occurrence of events in a multivariate Hawkes process using Granger causal inference. The primary objective is to leverage causal inference techniques to analyze and predict event occurrences over time in various applications, such as financial market analysis, crime prediction, and healthcare.

## Skills Demonstrated

- **Causal Inference:** Techniques to determine causality and predict future events based on historical data.
- **Event Prediction:** Methods to predict the occurrence of events in time series data.
- **Hawkes Processes:** Statistical models for self-exciting point processes used to predict event times.

## Use Case

- **Financial Market Analysis:** Predicting market events and trends based on historical financial data.
- **Crime Prediction:** Forecasting crime occurrences to assist law enforcement and public safety initiatives.
- **Healthcare:** Predicting health-related events and disease outbreaks to enhance preventive care and resource allocation.

## Components

### 1. Data Collection and Preprocessing

Collect and preprocess event data to ensure it is clean, consistent, and ready for analysis.

- **Data Sources:** Historical event logs, financial data, crime records, healthcare reports.
- **Techniques Used:** Data cleaning, normalization, feature extraction, handling missing data.

### 2. Hawkes Processes

Develop models based on Hawkes processes to capture the temporal dependencies between events.

- **Techniques Used:** Parameter estimation, self-exciting processes.
- **Libraries/Tools:** tick, PyMC3.

### 3. Granger Causal Inference

Implement Granger causal inference to identify causal relationships between different event types.

- **Techniques Used:** Time series analysis, hypothesis testing.
- **Libraries/Tools:** statsmodels.

### 4. Event Prediction

Build prediction models to forecast future events based on the Hawkes process and causal inference analysis.

- **Techniques Used:** Predictive modeling, time series forecasting.
- **Libraries/Tools:** scikit-learn, tick.

### 5. Evaluation and Validation

Evaluate the performance of the event prediction models using appropriate metrics.

- **Metrics Used:** Precision, Recall, F1-score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).

## Project Structure

```
hawkes_event_prediction/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── hawkes_processes.ipynb
│   ├── granger_causal_inference.ipynb
│   ├── event_prediction.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── hawkes_processes.py
│   ├── granger_causal_inference.py
│   ├── event_prediction.py
│   ├── evaluation.py
├── models/
│   ├── hawkes_model.pkl
│   ├── event_prediction_model.pkl
├── README.md
├── requirements.txt
├── setup.py
```

## Getting Started

### Prerequisites

- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hawkes_event_prediction.git
   cd hawkes_event_prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place raw event data files in the `data/raw/` directory.
2. Run the data preprocessing script to prepare the data:
   ```bash
   python src/data_preprocessing.py
   ```

### Running the Notebooks

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open and run the notebooks in the `notebooks/` directory to preprocess data, develop models, and evaluate the system:
   - `data_preprocessing.ipynb`
   - `hawkes_processes.ipynb`
   - `granger_causal_inference.ipynb`
   - `event_prediction.ipynb`
   - `evaluation.ipynb`

### Training and Evaluation

1. Train the Hawkes process models:
   ```bash
   python src/hawkes_processes.py --train
   ```

2. Train the event prediction models:
   ```bash
   python src/event_prediction.py --train
   ```

3. Evaluate the models:
   ```bash
   python src/evaluation.py --evaluate
   ```

## Results and Evaluation

- **Hawkes Processes:** Successfully modeled the temporal dependencies between events.
- **Causal Inference:** Identified causal relationships between different event types.
- **Event Prediction:** Developed models that accurately predict future events with high precision and recall.

## Contributing

We welcome contributions from the community. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and supporters of this project.
- Special thanks to the causal inference and machine learning communities for their invaluable resources and support.
```
