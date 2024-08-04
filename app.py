from flask import Flask, request, render_template
import pandas as pd
import joblib


class CombinedPipeline:
    def __init__(self, clf, reg1, reg2, reg3):
        self.clf = clf
        self.reg1 = reg1
        self.reg2 = reg2
        self.reg3 = reg3

    def fit(self, X, y_class, y_reg1, y_reg2, y_reg3):
        self.clf.fit(X, y_class)
        X_filtered = X[y_class == 1]  # Filter entries where class is 'Current' (assuming 1 represents 'Current')
        self.reg1.fit(X_filtered, y_reg1[y_class == 1])
        self.reg2.fit(X_filtered, y_reg2[y_class == 1])
        self.reg3.fit(X_filtered, y_reg3[y_class == 1])
        return self

    def predict(self, X):
        y_class_pred = self.clf.predict(X)
        X_filtered = X[y_class_pred == 1]
        y_reg1_pred = self.reg1.predict(X_filtered) if len(X_filtered) > 0 else [None]
        y_reg2_pred = self.reg2.predict(X_filtered) if len(X_filtered) > 0 else [None]
        y_reg3_pred = self.reg3.predict(X_filtered) if len(X_filtered) > 0 else [None]
        return y_class_pred, y_reg1_pred, y_reg2_pred, y_reg3_pred

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('Pipeline.pkl')

# List of selected important features
selected = ['BorrowerAPR', 'MonthlyLoanPayment', 'LoanOriginalAmount', 
    'BorrowerRate', 'LoanNumber','LP_ServiceFees','EstimatedEffectiveYield','EstimatedReturn', 'LoanCurrentDaysDelinquent', 'StatedMonthlyIncome']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request and validate
        data = {}
        for feature in selected:
            value = request.form.get(feature)
            if value is None or not value.replace('.', '', 1).isdigit():
                return f"Invalid input for {feature}", 400
            data[feature] = float(value)
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Predict using the loaded pipeline
        y_class_pred, y_reg1_pred, y_reg2_pred, y_reg3_pred = pipeline.predict(df)
        
        # Prepare the response
        return render_template('result.html', 
                               classification=y_class_pred[0],
                               loan_tenure=y_reg1_pred[0],
                               emi=y_reg2_pred[0],
                               ela=y_reg3_pred[0])
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

