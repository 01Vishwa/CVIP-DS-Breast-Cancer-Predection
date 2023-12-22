# -*- coding: utf-8 -*-
"""Breast Cancer Predection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vcMHUN1ohP97yKwrKpCe8DUBIVM4mhgJ
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,roc_curve

data=pd.read_csv("/content/breast-cancer.csv")

data.head()

data.info()

data.describe().T

data.columns

data.shape

data.isnull().sum()

duplicate_rows = data.duplicated().any()
duplicate_rows

cor_mat = data.corr()
cor_mat

mask = np.triu(np.ones_like(cor_mat, dtype=bool))
plt.figure(figsize=(28,28))
custom_palette = sns.color_palette("viridis", as_cmap=True)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cmap=custom_palette, cbar=True)
plt.show()



data.drop('id', axis=1, inplace=True)

for col in data.select_dtypes(include=np.number).columns:
    fig = px.histogram(data, x=col, title=f'Histogram for {col}')

    fig.update_layout(
        width=600,
        height=400,
    )

    fig.update_traces(marker_color='skyblue')

    fig.show()

data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
data.head()

worst_cols = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
              'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
              'fractal_dimension_worst']
data = data.drop(worst_cols, axis=1)

perimeter_area_cols = ['perimeter_mean', 'perimeter_se', 'area_mean', 'area_se']
data = data.drop(perimeter_area_cols, axis=1)

concavity_cols = ['concavity_mean', 'concavity_se', 'concave points_mean', 'concave points_se']
data = data.drop(concavity_cols, axis=1)

x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']

x.head()

scaler = StandardScaler()
X= scaler.fit_transform(x)

"""Logistics Regression"""

model=LogisticRegression()
model.fit(X,y)

y_pred=model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
conf_matrix = confusion_matrix(y, y_pred)

print("Metrics for Logistisc Regression Model:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

"""KNearestNeighbors"""

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

y_pred = knn_model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
conf_matrix = confusion_matrix(y, y_pred)
print("Metrics for KNN Model:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

"""Decision Tree"""

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x, y)

y_pred_dt = decision_tree_model.predict(x)

print("Metrics for Decision Tree Model:")
print("Accuracy:", accuracy_score(y, y_pred_dt))
print("Precision:", precision_score(y, y_pred_dt))
print("Recall:", recall_score(y, y_pred_dt))
print("F1 Score:", f1_score(y, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_dt))

"""Random Forest"""

random_forest_model = RandomForestClassifier(n_estimators=100)
random_forest_model.fit(x, y)

y_pred_rf = random_forest_model.predict(x)

print("\nMetrics for Random Forest Model:")
print("Accuracy:", accuracy_score(y, y_pred_rf))
print("Precision:", precision_score(y, y_pred_rf))
print("Recall:", recall_score(y, y_pred_rf))
print("F1 Score:", f1_score(y, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_rf))

"""Naive Bayes"""

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X, y)

y_pred = naive_bayes_model.predict(X)

accuracy = accuracy_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

print("\nMetrics for Naive Bayes Model:")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

"""XGB

"""

xgboost_classifier = xgb.XGBClassifier()
xgboost_classifier.fit(X,y)

y_pred = xgboost_classifier.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
conf_matrix = confusion_matrix(y, y_pred)

print("\nMetrics for XGB Model:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

model_names = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'XGBoost']

models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    xgb.XGBClassifier()
]

results = []

plt.figure(figsize=(10, 8))

for model, name in zip(models, model_names):
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    confusion = confusion_matrix(y, y_pred)
    results.append([name, accuracy, precision, confusion])

    y_proba = model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
    auc = roc_auc_score(y, y_proba[:, 1])

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classifiers')
plt.legend()
plt.show()

columns = ['Model', 'Accuracy', 'Precision', 'Confusion Matrix']
results_df = pd.DataFrame(results, columns=columns)
print(results_df)

model_names = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'XGBoost']

models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    xgb.XGBClassifier()
]

accuracies = []
for model in models:
    model.fit(X, y)
    accuracy = model.score(X, y)
    accuracies.append(accuracy)

fig = go.Figure(data=[
    go.Bar(
        x=model_names,
        y=accuracies,
        marker=dict(color=accuracies, colorscale='Viridis', showscale=True),
    )
])

fig.update_layout(
    title='Accuracy of Different Models',
    xaxis_title='Models',
    yaxis_title='Accuracy',
    width=800,
    height=500,
)

fig.show()

test_data_input = [
    float(input("Enter radius_mean: ")),
    float(input("Enter texture_mean: ")),
    float(input("Enter smoothness_mean: ")),
    float(input("Enter compactness_mean: ")),
    float(input("Enter symmetry_mean: ")),
    float(input("Enter fractal_dimension_mean: ")),
    float(input("Enter radius_se: ")),
    float(input("Enter texture_se: ")),
    float(input("Enter smoothness_se: ")),
    float(input("Enter compactness_se: ")),
    float(input("Enter symmetry_se: ")),
    float(input("Enter fractal_dimension_se: "))
]

test_data_df = pd.DataFrame([test_data_input], columns=[
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'smoothness_se', 'compactness_se', 'symmetry_se', 'fractal_dimension_se'
])

predicted = xgboost_classifier.predict(test_data_df)

if predicted[0] == 1:
    print("The patient is predicted to be a breast cancer patient.")
else:
    print("The patient is predicted not to be a breast cancer patient.")