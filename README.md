# Indian Road Accidents Analysis and Prediction

## Project Overview

Road accidents are a major public safety concern. Analyzing accident-related data can help identify patterns associated with road incidents and passenger outcomes.

This project explores Indian road accident data using Python, focusing on data exploration, preprocessing, visualization, class imbalance handling, and machine learning classification.

The main objective is to investigate the available accident and passenger information and develop machine learning models to predict whether a passenger survived an accident.

## Project Objectives

* Explore the dataset and understand its structure.
* Perform Exploratory Data Analysis (EDA).
* Identify missing values and duplicate records.
* Handle missing values in selected columns.
* Detect and cap outliers using the Interquartile Range (IQR) method.
* Visualize accident-related variables and their distributions.
* Prepare the data for machine learning.
* Address class imbalance using SMOTE.
* Train and compare multiple classification algorithms.
* Evaluate model performance using classification metrics.
* Identify important features associated with the model's predictions.
* Save the trained XGBoost model for potential future use.

## Dataset

The project uses a passenger-level Indian road accident dataset.

The dataset is expected to be available as:

`indian_road_accidents_passenger_level.csv`

Some of the variables explored in the notebook include:

* `speed_at_impact_kmph`: Speed at the time of impact.
* `passenger_age_years`: Passenger age.
* `emergency_response_time_min`: Emergency response time in minutes.
* `weather_condition`: Weather conditions associated with the accident.
* `festival_name`: Festival information associated with the record.
* `survived`: Target variable representing passenger survival.

The dataset file is not included in this repository by default. Users should obtain the dataset from its authorized source and place it in the appropriate project directory before running the notebook.

## Tools and Technologies

* **Python** — Data analysis and machine learning.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical operations.
* **Matplotlib** — Data visualization.
* **Seaborn** — Statistical visualization.
* **Scikit-learn** — Data splitting, preprocessing, model training, and evaluation.
* **Imbalanced-learn** — Handling class imbalance with SMOTE.
* **XGBoost** — Gradient boosting classification.
* **Joblib** — Saving the trained model.
* **Google Colab / Jupyter Notebook** — Development environment.

## Project Workflow

### 1. Data Loading and Exploration

The dataset is loaded into a Pandas DataFrame. Initial exploration includes:

* Inspecting the first and last records.
* Checking dataset dimensions.
* Reviewing descriptive statistics.
* Examining data types.
* Identifying missing values and duplicates.
* Exploring unique values and the target variable distribution.

### 2. Data Cleaning and Preprocessing

The notebook includes several preprocessing steps:

* Replacing missing values in `festival_name` with `No Festival`.
* Identifying outliers in selected numerical variables using the IQR method.
* Capping detected outliers at the calculated lower and upper bounds.
* Encoding selected categorical variables using one-hot encoding.
* Removing selected identifiers and outcome-related columns from model features.

### 3. Exploratory Data Analysis

Visualizations are used to explore the dataset, including:

* Distribution of impact speed.
* Distribution of passenger age.
* Distribution of weather conditions.
* Emergency response time in relation to passenger survival.
* Boxplots comparing numerical variables before and after outlier capping.

These visualizations help examine patterns and distributions in the available data.

### 4. Handling Class Imbalance

The notebook uses the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

SMOTE is applied to the training data to increase representation of the minority class before model training.

The test set remains separate for model evaluation.

### 5. Machine Learning Models

The notebook experiments with the following classification algorithms:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* XGBoost

The target variable is `survived`.

### 6. Model Evaluation

The notebook evaluates the models using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics provide different perspectives on classification performance, particularly when the target classes are imbalanced.

Actual performance values should be taken from the notebook outputs after running the experiments.

### 7. Feature Importance

The project uses XGBoost feature importance to identify the features that contribute most to the trained model's predictions.

Feature importance indicates how the model uses the available variables; it does not establish causation.

### 8. Model Saving

The trained XGBoost classifier is saved using Joblib as:

`traffic_accident_xgb_model.pkl`

This file can be used in a future inference workflow, provided the same feature preparation and column structure are maintained.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Indian-Road-Accidents-Analysis-and-Prediction.git
```

Navigate to the project directory:

```bash
cd Indian-Road-Accidents-Analysis-and-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

## How to Run

1. Clone or download the repository.
2. Install the required Python libraries.
3. Obtain the dataset and place it in the expected location.
4. Open the notebook in Jupyter Notebook or Google Colab.
5. Update the dataset path if necessary.
6. Run the notebook cells in order.
7. Review the visualizations, model evaluation reports, and feature importance chart.

## Project Structure

```text
Indian-Road-Accidents-Analysis-and-Prediction/
│
├── traffic_accidents.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── indian_road_accidents_passenger_level.csv
```

The dataset and trained model files may be excluded from the repository if they are large, restricted, or not intended for public distribution.

## Limitations

* Model performance depends on dataset quality, representativeness, and feature availability.
* Outlier capping may change the distribution of numerical variables.
* SMOTE generates synthetic training examples and does not replace the need for careful evaluation.
* Feature importance should not be interpreted as proof of causal relationships.
* Predictions should not be treated as definitive assessments of real-world passenger survival.
* The saved model requires consistent preprocessing and feature ordering during inference.

## Future Improvements

* Build a reusable preprocessing and modeling pipeline.
* Compare models using cross-validation.
* Tune hyperparameters to improve generalization.
* Evaluate precision, recall, and F1-score alongside accuracy.
* Investigate data leakage and feature selection more thoroughly.
* Add model explainability using SHAP.
* Develop an interactive dashboard to explore accident patterns.
* Build a simple web application for model predictions after validating the model.
* Document dataset provenance and add verified performance results.

## Disclaimer

This project is intended for educational and analytical purposes. Its predictions are not a substitute for professional emergency assessment or official road safety analysis.

## Author

menna waleed

Data Analytics | Python | Exploratory Data Analysis | Machine Learning

GitHub: https://github.com/menna-waleed
