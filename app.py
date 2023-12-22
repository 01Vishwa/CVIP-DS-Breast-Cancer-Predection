from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

app = Flask(__name__)

data = pd.read_csv("breast_cancer.csv")

data.drop('id', axis=1, inplace=True)

worst_cols = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
              'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
              'fractal_dimension_worst']
data = data.drop(worst_cols, axis=1)

perimeter_area_cols = ['perimeter_mean', 'perimeter_se', 'area_mean', 'area_se']
data = data.drop(perimeter_area_cols, axis=1)

concavity_cols = ['concavity_mean', 'concavity_se', 'concave points_mean', 'concave points_se']
data = data.drop(concavity_cols, axis=1)

data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)

x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']

scaler = StandardScaler()
X= scaler.fit_transform(x)

xgboost_classifier = xgb.XGBClassifier()
xgboost_classifier.fit(X,y)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        values = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean']),
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se']),
        ]
        print("Form Data:", values)
        print("Scaled Input:", values)

        try:
            prediction = xgboost_classifier.predict([values])
            print("Prediction:", prediction[0])  # Printing the prediction value

            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            error_message = f"Prediction Error: {str(e)}"
            print(error_message)  # Print the error message
            return render_template('result.html', prediction=error_message)

if __name__ == '__main__':
    app.run(debug=True)
