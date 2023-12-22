# CVIP-DS-Breast-Cancer-Predection
Breast Cancer Prediction Project
This repository contains code for a breast cancer prediction project developed during the internship at Coder's Cave. The project utilizes various machine learning algorithms and achieves high accuracy in predicting breast cancer.

Model Performance
The following table displays the performance metrics of different machine learning models:

Model	Accuracy	Precision	Confusion Matrix
Logistic Regression	0.940246	0.940109	[[342, 15], [19, 193]]
KNN	0.954306	0.954887	[[351, 6], [20, 192]]
Decision Tree	1.000000	1.000000	[[357, 0], [0, 212]]
Random Forest	1.000000	1.000000	[[357, 0], [0, 212]]
Naive Bayes	0.903339	0.903511	[[339, 18], [37, 175]]
XGBoost	1.000000	1.000000	[[357, 0], [0, 212]]
For the final implementation, XGBoost was chosen as the algorithm for breast cancer prediction due to its high accuracy and performance.

Flask UI for Breast Cancer Prediction
The Flask user interface (UI) was developed to allow users to interact with the trained XGBoost model for breast cancer prediction. The interface provides a user-friendly way to input relevant data and obtain predictions regarding the likelihood of breast cancer.

To run the Flask application:

Clone this repository.
Install the required dependencies using pip install -r requirements.txt.
Run the Flask app using python app.py.
Access the application via the provided URL.
Feel free to explore and contribute to this breast cancer prediction project!
