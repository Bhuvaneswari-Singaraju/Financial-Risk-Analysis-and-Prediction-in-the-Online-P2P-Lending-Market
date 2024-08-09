# Financial Risk Analysis and Prediction in the Online P2P Lending Market

## Overview

This project aims to analyze financial risks and predict outcomes in the online peer-to-peer (P2P) lending market using machine learning techniques. It involves data preprocessing, feature engineering, and building predictive models.

## Data

The dataset contains information on loan applications, borrower details, and loan performance. Key columns include:

- `BorrowerAPR`: Borrower's Annual Percentage Rate
- `MonthlyLoanPayment`: Monthly loan payment
- `LP_NonPrincipalRecoverypayments`: Non-principal recovery payments
- `LenderYield`: Yield for the lender
- `LoanStatus`: Status of the loan (e.g., Fully Paid, Charged Off)
- `LoanOriginalAmount`: Original loan amount
- `StatedMonthlyIncome`: Stated monthly income of the borrower

## Setup

1. **Create and activate a virtual environment:**

    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

2. **Install dependencies**

## Loading Packages and Data

1. **Import necessary libraries:**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```
2. **Load the dataset:**

    ```python
    df = pd.read_csv('prosperLoanData.csv')
    ```
## Data Structure and Content

1. **Check the structure of the dataset:**

    ```python
    print(df.head())
    print(df.info())
    print(df.describe())
    ```
2. **Examine the data types and missing values:**

    ```python
    print(df.dtypes)
    print(df.isnull().sum())
    ```

3. **Treat the missing values separately for both the numerical and categorical data with suitable strategies.**

4. **Check for the Outliers by any of the methods (In this project, Boxplot is used to detect):**

    ```python
    sns.boxplot(data = df, x = i)
    plt.show()
    ```

5. **Treat the outliers (selective) using IQR Method and then visualize them using boxplots after the treatments**

## Classification
### Binary Classification

In this section, we will create a binary classification target variable `Status` based on the `ClosedDate` and `LoanCurrentDaysDelinquent` columns.

1. **Creating Binary Classification Target Variable:**

    - **Status based on `ClosedDate`:** If `ClosedDate` is `NaN` (i.e., the loan is not closed), assign a value of `1`. Otherwise, assign `0`.

    - **Status based on `LoanCurrentDaysDelinquent`:** If `LoanCurrentDaysDelinquent` is greater than `180` days, assign a value of `1` (indicating high delinquency). Otherwise, assign `0`.

2. **Verify the Distribution of the Binary Target Variable:**

    Check the distribution of the `Status` variable to understand the balance between the classes:

    ```python
    # Display the value counts of the binary target variable 'Status'
    print(df['Status'].value_counts())
    ```

    This output will provide a count of each class in the `Status` column, to assess whether the classes are balanced or imbalanced.

## Data Encoding

Before performing feature engineering, it's crucial to encode categorical variables into numerical format. This allows machine learning algorithms to effectively process and learn from the data.

2. **Label Encoding:**

    - **Convert Labels to Numbers:** For ordinal categorical variables, use label encoding to assign numerical values to categories.

    ```python
    from sklearn.preprocessing import LabelEncoder

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    df['LoanStatus'] = label_encoder.fit_transform(df['LoanStatus'])
    ```
## Feature Engineering

After encoding categorical variables, we proceed with feature engineering to enhance the predictive power of our model.

1. **Create Target Variables:**

    ```python
    X = df.drop('LoanStatus', axis=1)
    y = df['LoanStatus']
    ```

    This step prepares the feature set (`X`) and target variable (`y`) for model building.

**Here, create a copy of the dataframe df i.e., df_copy and find the important features of the target variable by running the MI scores feature engineering method to the copy of the df.** 

2. **Feature Importance:**

    - **Find Mutual Information Scores:** Use mutual information scores to identify the importance of features with respect to the target variable.

    ```python
    from sklearn.feature_selection import mutual_info_classif

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    ```

    Mutual information scores help assess the dependency between features and the target variable, guiding feature selection and engineering.

**After finding the important features, we can remove some unwanted and less important features of the target variable.**


4. **Save the data which is in the original format and convert it to csv (dataset after the changes made like dropping some features and etc.,) for the Regression task.**


## Data Encoding

Data encoding is now done for the original dataframe df. Label Encoding is used here.

## Exploratory Data Analysis (EDA)

1. **Visualize the distribution of key variables:**

    ```python
    plt.figure(figsize = (6,6))
    plt.hist(df[col], edgecolor = 'black')
    plt.title(col)
    plt.show()
    ```

    ```python
    plt.figure(figsize = (6,6))
    pie = df[col].value_counts()
    wedges, texts = plt.pie(pie, startangle = 90, colors = sns.color_palette('coolwarm'))
    plt.title(col)
    plt.show()
    ```

2. **Explore relationships between variables:**

    ```python
    numcols1 = ['BorrowerAPR', 'BorrowerRate', 'DebtToIncomeRatio', 'MonthlyLoanPayment']
    sns.pairplot(df[numcols1])
    plt.suptitle("Scatter Plots of Important Numerical Features", y=1.02)
    plt.show()
    ```

3. **Visualize the Correlation coefficients:**

    Heatmap is generally used to visulaize the correlation coefficents.

4. **Multivariate Data Analysis:**

- Dimensionality Reduction using PCA
PCA is imported from the 'sklearn.decomposition' module, which is used for the dimensionality reduction.

- And the visualization is done using scatter plot and also the 3D plot.

## Feature Engineering

### Resampling using SMOTE (Synthetic Minority Over Sampling Technique)

**SMOTE** is done by generating synthetic samples for the minority class.
- It identifies the imbalance.
- Focus on the minority.
  - Creates synthetic samples just for the minority data points and generates the new ones similar to them.
- Increase minority.
  - By adding these, SMOTE balances the data, giving the model a better chance to learn the minority class.

  1. Feature Selection
  2. Finding the Mutual Information
  3. Normalization, Scaling and PCA (Principal Component Analysis)

## Model Building

- The model building is done by finding the accuracy_score, confuxion_matrix, classification_report and then we choose the best model according the results.

- Then the Cross-Validation is also doen to each and every model to improve the Model Performance.

- Logistic Regression, Decision Tree Classification, Naive Bayes Classification, SVM (Support Vector Machine) are the models.

**Model Performance Analysis - (Classification)**
1. **Logistic Regression**
Accuracy: 0.97
Precision: High for both classes, especially for class 0 (0.99).
Recall: Very high for class 0 (0.98), good for class 1 (0.91).
F1-Score: High for both classes, indicating balanced performance.
Cross-Validation Mean Accuracy: 0.97

2. **Decision Tree Classification**
Accuracy: 0.91
Precision: High for class 0 (0.97), moderate for class 1 (0.59).
Recall: High for class 0 (0.93), moderate for class 1 (0.78).
F1-Score: High for class 0 (0.95), moderate for class 1 (0.67).
Cross-Validation Mean Accuracy: 0.93

3. **Naive Bayes Classification**
Accuracy: 0.90
Precision: High for class 0 (0.96), lower for class 1 (0.55).
Recall: High for class 0 (0.92), moderate for class 1 (0.73).
F1-Score: High for class 0 (0.94), moderate for class 1 (0.63).
Cross-Validation Mean Accuracy: 0.90

4. **Support Vector Machine (SVM)**
Accuracy: 0.99
Precision: Very high for both classes, especially for class 0 (1.00).
Recall: Very high for both classes, especially for class 0 (0.99).
F1-Score: Very high for both classes, indicating excellent performance.
Cross-Validation Mean Accuracy: 0.99


**Logistic Regression and SVM show the highest overall accuracy and balanced performance across precision, recall, and F1-score.**


## Regression

## Loading the Dataset

The dataset which is previously saved to csv for the Regression task is used here.

## Data Structure and Content

1. **Check the structure of the dataset:**

    ```python
    print(dfreg.head())
    print(dfreg.info())
    print(dfreg.describe())
    ```
2. **Examine the data types and missing values:**

    ```python
    print(dfreg.dtypes)
    print(dfreg.isnull().sum())
    ```

3. **Treat the missing values separately for both the numerical and categorical data with suitable strategies.**
 

4. **Check for the duplicate values. If any, then remove them.**

## Creting new Target variables for the Borrower Rate:

**Loan Tenure**

```
LoanTenure = (𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑦𝑒𝑎𝑟 − 𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑦𝑒𝑎𝑟) 𝑥 12 − (𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡𝑒_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑚𝑜𝑛𝑡ℎ − 𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑚𝑜𝑛𝑡ℎ)
```

**Equated Monthly Installments (EMI)**

```
Tenure ---> LoanTenure
Principle repayment ---> LP_CustomerPrinciplePayments
Interest ---> BorrowerRate
```
**EMI = P × r × (1 + r) ^ n / ((1 + r) ^ n – 1)**

**Eligible Loan Amount (ELA)**

```
Components of ELA:

A: “AppliedAmount” ---> LoanOriginalAmount
R: “Interest” ---> BorrowerRate
N: “LoanTenure” ---> LoanTenure
I: “IncomeTotal” ---> StatedMonthlyIncome
```
**After creation of new target variables, save the dataframe dfreg to csv file "DataForPipeline.csv" which is further used for the Combined Pipeline creation.**

Now after the creation of the target variables, we can plot the WOE.

## Model Building for the three target variables 

- Linear Regression, Lasso Regression, Ridge Regression are done and the results are compared based on the R2_Scores, Mean Squared Errors.

- These results are stored in a dataframe which can be accessed easily.


        Model	      Target Variable	   RMSE	           R2

0	Linear Regression	LoanTenure       4.742087   	0.804918
1	Linear Regression	EMI              147.745425	    0.925904
2	Linear Regression	ELA	             15996.703386	0.926250
3	Lasso Regression	LoanTenure	     5.613015	    0.726681
4	Lasso Regression	EMI	             157.365993	    0.915940
5	Lasso Regression	ELA	             15997.554034	0.926242
6	Ridge Regression	LoanTenure       4.793496	    0.800666
7	Ridge Regression	EMI	             148.489202 	0.925156
8	Ridge Regression	ELA	             16000.234253	0.926218

The above are the Results of the each model.

- The cross validations are done for each model and the final model performances are evaluated.

## Model Performance Analysis - (Regression)

### Linear Regression
- *LoanTenure*:
  - *RMSE*: 4.742087
  - *R²*: 0.804918
  - *Cross-Validation R² Scores*: [0.78808227, 0.80266106, 0.79677897, 0.79702707, 0.79222777]
  - *Mean Cross-Validation R² Score*: 0.7953554279532579

- *EMI*:
  - *RMSE*: 147.745425
  - *R²*: 0.925904
  - *Cross-Validation R² Scores*: [0.92395884, 0.92683246, 0.92638722, 0.92466062, 0.92319102]
  - *Mean Cross-Validation R² Score*: 0.9250060329568107

- *ELA*:
  - *RMSE*: 15996.703386
  - *R²*: 0.926250
  - *Cross-Validation R² Scores*: [0.9281865, 0.93657764, 0.92695647, 0.93429285, 0.92861042]
  - *Mean Cross-Validation R² Score*: 0.930924778113749

### Lasso Regression
- *LoanTenure*:
  - *RMSE*: 5.613015
  - *R²*: 0.726681

- *EMI*:
  - *RMSE*: 157.365993
  - *R²*: 0.915940

- *ELA*:
  - *RMSE*: 15997.554034
  - *R²*: 0.926242

### Ridge Regression
- *LoanTenure*:
  - *RMSE*: 4.793496
  - *R²*: 0.800666

- *EMI*:
  - *RMSE*: 148.489202
  - *R²*: 0.925156

- *ELA*:
  - *RMSE*: 16000.234253
  - *R²*: 0.926218


- *Linear Regression*:
  - Performs best overall with high R² values and lower RMSE across all target variables.
  - *LoanTenure* has a mean cross-validation R² score of 0.795, indicating good generalization.
  - *EMI* and *ELA* have very high R² values (0.925 and 0.926 respectively) with consistent cross-validation scores.

- *Lasso Regression*:
  - Generally has slightly higher RMSE and lower R² compared to Linear Regression.
  - Still performs reasonably well but is less accurate for *LoanTenure* with an R² of 0.726.

- *Ridge Regression*:
  - Similar performance to Linear Regression but with slightly higher RMSE and lower R² values.
  - Offers a balance between model simplicity and performance.

- **Overall, *Linear Regression* shows the best performance, especially for predicting *EMI* and *ELA*, with high R² scores and low RMSE, making it the preferred model for these targets.**


## Pipeline


In this project, we use the scikit-learn `Pipeline` for integrating multiple stages of the machine learning workflow, including data preprocessing and model fittiny:

1. **Integration of Multiple Models:**
   - The project involves both classification (`Status` prediction) and regression tasks (`LoanTenure`, `EMI`, `ELA` predictions). The `Pipeline` allows us to seamlessly integrate these tasks into a single workflow.

2. **Consistent Data Transformation:**
   - By defining a `Pipeline`, we ensure that all data preprocessing steps, such as scaling and handling missing values, are applied consistently to both the training and testing datasets. This prevents data leakage and ensures robust model eval **consistency.

Overall, the `Pipeline` enhances efficiency, maintains data integrity throughout the workflow, and prepares the models for deployment in real-wor**ld applications.


**A combined Pipeline is created which is then fit into the training data and then the predictions are found with the test data.**

The final results are then saved to a dataframe and they are easily verified.

### Model Performance Analysis - (Pipeline)

#### Classification Model:
- **Accuracy:** 95.17%
- **Precision:** 84.51%
- **Recall:** 72.28%
- **F1-Score:** 77.92%

#### Regression Models:

##### Loan Tenure Regression:
- **Mean Absolute Error (MAE):** 2.0411
- **Mean Squared Error (MSE):** 9.9264
- **R-squared (R2):** 0.7993

##### EMI Regression:
- **MAE:** 54.6622
- **MSE:** 6758.2471
- **R2:** 0.9327

##### ELA Regression:
- **MAE:** 11529.8422
- **MSE:** 612210976reas for potential improvement.


### Overall Analysis:
- **Classification Model:** The model effectively predicts loan approval status with high accuracy, precision, recall, and F1-score, indicating robust performance.
- **Regression Models:** 
  - **EMI Regression:** Shows excellent predictive accuracy with low MAE, MSE, and high R2 score, indicating precise monthly payment predictions.
  - **Loan Tenure Regression:** Performs well with moderate errors and explains a substantial portion of variance in loan tenure predictions.
  - **ELA Regression:** Demonstrates higher errors compared to EMI and Loan Tenure, indicating potential for improvement, though still explains a significant amount of variance in expected loan amounts.


## Saving the Pipeline using Joblib

**`joblib.dump` to save the pipeline object and then `joblib.load` to load it back is correct for persisting your trained model.**

### Saving and Loading a Machine Learning Pipeline

- **Saving:** `joblib.dump(pipeline, 'CombinedPipeline.pkl')` saves the entire pipeline with trained models and preprocessing steps.

- **Loading:** `loaded_pipeline = joblib.load('CombinedPipeline.pkl')` loads the pipeline back into memory.

- **Predicting:** Use `loaded_pipeline.predict(X_test)` to apply the saved models (`clf`, `reg1`, `reg2`, `reg3`) on new data (`X_test`).

This method enables efficient reuse of trained models without retraining, suitable for deployment and further analysis.


## Deployment

- Create an `app.py` using Flask and load the pickle file which is saved earlier. 

- `index.html` and `result.html` are also created according to the requirements we want and then these are called into the `app.py` file.

- **To run the whole project, `app.py` should be run to get the end website where the user enters the features which are given on the page (features which are earlier selected in the Pipeline.ipynb) and then click on `Submit` button which goes to the result page showing the results of both classification and regression simultaneously on the same screen.**

- The validation is also there where the user is only allowed to give the input in nuumbers. 

- The user need to fill in all the features otherwise, he will not be able to go to the result page and also it will appear `Please fill out this field!` on the feature where the user didn't fill the input.

- In the result page, after obtaining the output, there is an option called `Go Back` where the user can go back to the index page and again check for some other details which is more feasible because there is no need to everytime run the code for every session.


## Specifications

**Python 3.10.11**

Name: **scikit-learn**
Version: **1.2.2**

Name: **pandas**
Version: **2.2.2**

Name: **numpy**
Version: **1.23.5**

Name: **joblib**
Version: **1.4.2**

Name: **Flask**
Version: **3.0.3**

# **Financial-Risk-Analysis-and-Prediction-online-P2P-Lending-Market**
