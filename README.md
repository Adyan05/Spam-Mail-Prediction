# Spam Mail Prediction

## Overview

This project focuses on predicting whether an email is spam or not using machine learning techniques. The goal is to build a model that accurately classifies emails based on their content, metadata, and other relevant features. This project can be a valuable tool for email service providers and individuals looking to filter out unwanted spam messages.

## Features

- **Data Cleaning and Preprocessing:** Handling missing data, text preprocessing, and converting categorical variables.
- **Exploratory Data Analysis (EDA):** Visualization of the dataset to understand the distribution of spam vs. non-spam emails.
- **Feature Engineering:** Extraction of key features like word frequency, presence of certain keywords, and metadata.
- **Modeling:** Implementation of various machine learning models including Naive Bayes, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning:** Optimization of model parameters to improve classification accuracy.
- **Model Evaluation:** Comparison of models using metrics like accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python 3.x**
- **Pandas** - For data manipulation and analysis
- **NumPy** - For numerical computations
- **Scikit-learn** - For machine learning algorithms and model evaluation, including:
  - `train_test_split` - For splitting the dataset into training and testing sets
  - `TfidfVectorizer` - For converting text data into numerical features
  - `LogisticRegression` - For building the classification model
  - `accuracy_score` - For evaluating model performance
  - `LabelEncoder` - For encoding categorical labels into numerical values


## Installation

To run this project using Google Colab, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Spam-Mail-Prediction.git

2. Upload the cloned repository to your Google Drive.
3. Open Google Colab and run the code.

## Alternative Installation

To run this project on Jupyter Notebook, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Adyan05/Spam-Mail-Prediction.git
2. Navigate to the project directory.
3. Install the required libraries.
4. Open and run the Jupyter Notebook
   ```bash
   !jupyter notebook SpamMailPredictionProject.ipynb
## Acknowledgments

- Special thanks to the creators of the open-source libraries used in this project, including [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), and [Scikit-learn](https://scikit-learn.org/).
- The dataset used for this project can be sourced from various open datasets, such as the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/).
- This project is referenced from (https://www.youtube.com/watch?v=rxkGItX5gGE&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=19).
