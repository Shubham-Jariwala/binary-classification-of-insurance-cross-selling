# Binary Classification of Insurance Cross Sell Prediction

## Introduction

The goal of this project is to predict which customers will respond positively to an automobile insurance offer. This is a binary classification problem where the target variable is 'Response'. The evaluation metric is the Area Under the ROC Curve (AUC).

## Data

The dataset consists of training and test data, with various features describing customers and their insurance history. The target variable `Response` is imbalanced, with a significantly lower number of positive responses.

## Approach

1.  **Data Loading and Initial Look**: Load the training and test data and perform a preliminary inspection of their shapes and the distribution of the target variable.
2.  **Preprocessing**:
    *   **Downsampling**: Due to the imbalanced target variable, downsampling of the majority class is performed to create a more balanced training dataset.
    *   **Feature Engineering**: Categorical features (`Gender`, `Vehicle_Age`, `Vehicle_Damage`) are encoded. `Gender` and `Vehicle_Damage` are one-hot encoded (though the code uses integer mapping), and `Vehicle_Age` is treated as an ordinal feature and mapped to integers. Other features are converted to more memory-efficient data types.
3.  **Exploratory Data Analysis (EDA)**:
    *   **Mutual Information Importance**: Mutual information is calculated between features and the target variable to assess their potential relevance for the model. A noise feature is added as a baseline for comparison.
    *   **Statistical Properties**: Basic statistical properties of the data are examined (commented out in the provided notebook).
4.  **Modeling**:
    *   **Model Selection**: An XGBoost classifier is chosen for the prediction task.
    *   **Training**: The XGBoost model is trained on the preprocessed and downsampled training data. Early stopping is used to prevent overfitting.
    *   **Cross-validation (commented out)**: A cross-validation function is defined but commented out in the provided notebook.
    *   **LightGBM (commented out)**: Code for training a LightGBM model with specific parameters is present but commented out.
    *   **Random Forest (commented out)**: Code for training and evaluating a Random Forest model is present but commented out.
5.  **Evaluation**:
    *   The trained XGBoost model is evaluated on a held-out test set using the AUC metric.
    *   ROC curve plotting code is present but commented out.
6.  **Submission**:
    *   Predictions are generated for the test data using the trained XGBoost model.
    *   A submission file in the specified format (`submission.csv`) is created with the predicted probabilities.

## Dependencies

The project uses the following libraries:

*   `numpy`
*   `pandas`
*   `datetime`
*   `matplotlib.pyplot`
*   `seaborn`
*   `sklearn`
*   `lightgbm`
*   `xgboost`
*   `kagglehub`

## How to Run

1.  Ensure you have the necessary dependencies installed.
2.  Download the dataset from the specified Kaggle competition (`playground-series-s4e7`).
3.  Run the notebook cells sequentially.

**Note**: Some code cells related to cross-validation, LightGBM, and Random Forest are commented out in the provided notebook. To use them, uncomment the respective cells.
