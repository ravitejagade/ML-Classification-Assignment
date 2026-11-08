Machine Learning Classification Assignment
Project Overview

This project implements a Machine Learning Classification Dashboard developed as part of the BITS Pilani M.Tech AI/ML Assignment 2.
Classification Models

The following five machine learning algorithms are implemented and compared:

    Logistic Regression
    Decision Tree
    K-Nearest Neighbors (KNN)
    Gaussian Naive Bayes
    Support Vector Machine (SVM)

Dataset

The project uses a classification dataset with multiple numerical features.
Model Evaluation

Models are evaluated using the following performance metrics:

    Accuracy
    Precision
    Recall
    F1 Score
    AUC
    Matthews Correlation Coefficient (MCC)

Application

An interactive Streamlit dashboard allows users to:

    Select classification models.
    Upload test datasets in CSV format.
    View dataset previews.
    Evaluate model performance.
    Compare classification results.
    View confusion matrices and classification reports.

Technologies Used

    Python
    Scikit-learn
    Pandas
    Streamlit
    Matplotlib
    Seaborn

Project Files

    app.py – Streamlit dashboard
    train_model.py – Model training script
    dataset.csv – Training dataset
    test_data.csv – Testing dataset
    metrics.csv – Model evaluation metrics
    feature_columns.json – Feature configuration
    Model .pkl files – Trained classification models
