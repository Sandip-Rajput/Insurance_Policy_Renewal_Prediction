# Insurance Policy Renewal Prediction using Machine Learning

## Overview

Insurance companies often face customer retention challenges when policyholders decide not to renew their insurance policies. This project aims to predict whether an existing customer will renew their policy next year using Machine Learning.

The model analyzes customer demographics, policy details, claim history, payment behavior, and satisfaction metrics to identify customers who are at risk of not renewing their policies.

---

## Business Problem

Insurance companies need to identify customers who are likely to discontinue their policies.

Without prediction:

* Customer leaves unexpectedly
* Revenue loss occurs
* Retention opportunities are missed

With prediction:

* High-risk customers can be identified early
* Agents can proactively contact customers
* Personalized discounts and offers can be provided
* Customer retention can be improved

---

## Dataset Features

### Customer Information

* Age
* Gender
* City
* State
* Annual Income
* Occupation

### Policy Information

* Policy Type
* Premium Amount
* Sum Assured

### Customer History

* Years With Company
* Previous Renewals

### Claims Information

* Total Claims
* Claim Amount

### Customer Behaviour

* Late Payment Count
* Satisfaction Score
* Auto Renewal Enabled
* Agent Rating

### Target Variable

Renewal_Status_Next_Year

* Renewed
* Not Renewed

---

## Machine Learning Workflow

### Data Preprocessing

* Missing Value Handling
* Data Cleaning
* Incorrect Value Correction
* Feature Engineering

### Feature Engineering

Created:

* Policy_Year
* Policy_Month

from Policy_Start_Date.

### Encoding

Categorical features were transformed using:

* OneHotEncoder

### Feature Scaling

Applied:

* StandardScaler

### Target Encoding

Applied:

* LabelEncoder

### Model Building

Model Used:

* Random Forest Classifier

### Model Evaluation

* Train-Test Split
* Accuracy Evaluation
* Prediction Analysis

---

## Flask Web Application

A Flask-based web application was developed to perform real-time policy renewal prediction.

### Workflow

Customer Input

↓

OneHotEncoder

↓

StandardScaler

↓

Random Forest Model

↓

Prediction

↓

Renewed / Not Renewed

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Flask
* HTML
* CSS

---

## Future Improvements

### Bulk Prediction

Upload CSV file and predict multiple customers at once.

### Risk Probability

Display probability score.

Example:

Not Renewed Risk = 82%

### Customer Segmentation

Categorize customers into:

* Low Risk
* Medium Risk
* High Risk

### Dashboard

Interactive dashboard showing:

* Total Customers
* High Risk Customers
* City-wise Analysis
* Policy-wise Analysis

### Cloud Deployment

Deploy application on:

* Render
* Streamlit Cloud
* AWS

---

## Author

Sandip Rajput

Aspiring Data Scientist | Machine Learning Enthusiast

Ahmedabad, Gujarat
