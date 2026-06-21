from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
std = pickle.load(open("scaler.pkl", "rb"))
lb = pickle.load(open("label_encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

num_cols = [
    'Age', 'Annual_Income', 'Sum_Assured', 'Years_With_Company',
    'Total_Claims', 'Claim_Amount', 'Late_Payment_Count',
    'Previous_Renewals', 'Satisfaction_Score', 'Agent_Rating',
    'Policy_Year', 'Policy_Month'
]

cat_cols = [
    'Gender', 'City', 'State', 'Occupation',
    'Policy_Type', 'Auto_Renewal_Enabled'
]

# Dropdown values
gender_list = ["Male", "Female"]

city_list = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mumbai", "Delhi", "Pune", "Bangalore"]

state_list = ["Gujarat", "Maharashtra", "Delhi", "Karnataka", "Rajasthan", "Madhya Pradesh"]

occupation_list = ["Salaried", "Business", "Self Employed", "Student", "Retired"]

policy_type_list = ["Health", "Life", "Term", "Vehicle", "Travel"]

auto_renewal_list = ["Yes", "No"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        data = {
            "Age": [int(request.form["Age"])],
            "Gender": [request.form["Gender"]],
            "City": [request.form["City"]],
            "State": [request.form["State"]],
            "Annual_Income": [float(request.form["Annual_Income"])],
            "Occupation": [request.form["Occupation"]],
            "Policy_Type": [request.form["Policy_Type"]],
            "Sum_Assured": [float(request.form["Sum_Assured"])],
            "Years_With_Company": [int(request.form["Years_With_Company"])],
            "Total_Claims": [int(request.form["Total_Claims"])],
            "Claim_Amount": [float(request.form["Claim_Amount"])],
            "Late_Payment_Count": [int(request.form["Late_Payment_Count"])],
            "Previous_Renewals": [int(request.form["Previous_Renewals"])],
            "Satisfaction_Score": [float(request.form["Satisfaction_Score"])],
            "Auto_Renewal_Enabled": [request.form["Auto_Renewal_Enabled"]],
            "Agent_Rating": [float(request.form["Agent_Rating"])],
            "Policy_Year": [int(request.form["Policy_Year"])],
            "Policy_Month": [int(request.form["Policy_Month"])]
        }

        new_df = pd.DataFrame(data)

        new_num = new_df[num_cols].values
        new_cat = ohe.transform(new_df[cat_cols]).toarray()

        final_input = np.hstack((new_num, new_cat))
        final_df = pd.DataFrame(final_input, columns=columns)

        final_std = std.transform(final_df)

        pred = model.predict(final_std)
        prediction = lb.inverse_transform(pred)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        gender_list=gender_list,
        city_list=city_list,
        state_list=state_list,
        occupation_list=occupation_list,
        policy_type_list=policy_type_list,
        auto_renewal_list=auto_renewal_list
    )


if __name__ == "__main__":
    app.run(debug=True)