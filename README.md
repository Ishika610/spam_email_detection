# Spam Email Detection

Welcome to the Spam Email Detection project! This repository contains code and resources for detecting spam emails using machine learning techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to classify emails as spam or not spam (ham) using various machine learning algorithms. Accurate spam detection can help improve email filtering systems and enhance user experience by reducing the number of unwanted emails.

## Dataset

The dataset for the prediction is already provided with name spam.csv

### Features

The dataset includes various attributes extracted from the emails, such as:
- `word_freq_xxx`: Frequency of the word xxx in the email
- `char_freq_xxx`: Frequency of the character xxx in the email
- `capital_run_length_average`: Average length of sequences of capital letters
- `capital_run_length_longest`: Length of the longest sequence of capital letters
- `capital_run_length_total`: Total number of capital letters in the email

## Installation

To get started, clone this repository to your local machine:

```
git clone https://github.com/your-username/spam-email-detection.git
cd spam-email-detection
```

Install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Prepare the data for modeling by handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.

```python
python data_preprocessing.py
```

2. **Model Training**: Train various machine learning models to detect spam emails.

```python
python train_model.py
```

3. **Prediction**: Use the trained model to classify new emails as spam or ham.

```python
python predict.py
```

## Models

The project explores multiple classification algorithms:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## Results

The performance of each model is evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score. Below are the results for each model:

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 0.95     | 0.94      | 0.96   | 0.95     |
| Naive Bayes              | 0.93     | 0.92      | 0.94   | 0.93     |
| Support Vector Machine   | 0.97     | 0.96      | 0.97   | 0.97     |
| Decision Tree Classifier | 0.94     | 0.93      | 0.95   | 0.94     |
| Random Forest Classifier | 0.96     | 0.95      | 0.96   | 0.96     |
| Gradient Boosting Classifier | 0.97  | 0.96      | 0.97   | 0.97     |

