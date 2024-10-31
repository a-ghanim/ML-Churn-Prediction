import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
import utils as ut

# Check and print the API key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
print(f"GROQ_API_KEY: {api_key}")

# Set up the Groq client
client = Groq(api_key=api_key)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost_SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost_featureEngineered.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):

  input_dict = {
      'CreditScore': credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_products,
      'HasCreditCard': int(has_credit_card),
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      'Gender_Male': 1 if gender == 'Male' else 0,
      'Gender_Female': 1 if gender == 'Female' else 0,
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df):
  probabilities: dict[str, float] = {
      'XGBoost':
      float(f"{xgboost_model.predict_proba(input_df)[0][1]:.2g}"),
      'RandomForest':
      float(f"{random_forest_model.predict_proba(input_df)[0][1]:.2g}"),
      'K-Nearest Neighbors':
      float(f"{knn_model.predict_proba(input_df)[0][1]:.2g}"),
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown("### Model Probabilities")
  for model, prob in probabilities.items():
    st.write(f"{model}: {prob}")

  st.markdown(f"### Average Probability: {avg_probability:.2g}")
  return avg_probability

def groq_completion(prompt, model="gemma2-9b-it"):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return chat_completion

def explain_prediction(probability, input_dict, surname):
    high_risk_threshold = 0.4
    risk_level = "high" if probability > high_risk_threshold else "low"

    prompt = f"""
    You are an expert data scientist at a bank, specializing in explaining model-driven churn predictions to stakeholders.

    Based on the customer's profile, here is an interpretation of why {surname} might be at {risk_level} risk of churning.

    Customer's Profile:
    {input_dict}

    Top 5 Features Affecting Churn Probability:

    Feature       | Importance
    --------------|------------
    NumOfProducts | 0.323888
    IsActiveMember| 0.164146
    Age           | 0.109550
    Geography     | 0.091373
    Balance       | 0.052786

    Analyze these factors and generate a concise explanation of why this customer is at risk (if probability > 40%) or why they are less likely to churn (if probability <= 40%).

    Note: Keep the explanation clear, with no explicit mention of the model's inner workings or exact probabilities.
    """

    print("EXPLANATION PROMPT", prompt)

    raw_response = groq_completion(prompt)
    return raw_response.choices[0].message.content

def generate_email(avg_probability, input_dict, explanation, surname):
    prompt = f"""
    You are a customer loyalty manager at HS Bank. Your role is to ensure customers stay engaged with the bank by offering personalized incentives that align with their needs and preferences.

    You noticed that a valued customer, {surname}, might benefit from additional engagement initiatives. Below is an understanding of their profile:

    Customer Profile:
    {input_dict}

    Rationale:
    {explanation}

    Write a warm, engaging email to {surname} that highlights their importance to the bank and offers incentives tailored to their profile to enhance loyalty. Please include:

    - A personalized greeting and expression of appreciation for their business.
    - A list of 2-3 specific incentives (e.g., rewards, product discounts, or premium services) that align with their interests.
    - A friendly closing inviting them to reach out with any questions.

    Avoid mentioning churn probability or risk indicators. The goal is to make {surname} feel valued and appreciated, with clear benefits for staying engaged with the bank.
    """

    print("\nEMAIL PROMPT:", prompt)

    raw_response = groq_completion(prompt)
    return raw_response.choices[0].message.content

st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])

    print("Selected Customer ID", selected_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]

    print("Surname", selected_surname)

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id]
    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"].iloc[0]))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"].index(
                                    selected_customer["Geography"].iloc[0]))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer["Gender"].iloc[0] == "Male" else 1)

        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer["Age"].iloc[0]))

        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer["Tenure"].iloc[0]))

    with col2:

        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(
                                      selected_customer["Balance"].iloc[0]))

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"].iloc[0]))

        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard'].iloc[0]))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember'].iloc[0]))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"].iloc[0]))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)

    avg_probability = make_predictions(input_df)

    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer['Surname'])

    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

email = generate_email(avg_probability, input_dict, explanation,
                       selected_customer['Surname'])

st.markdown("---")
st.subheader("Personalized Email")
st.markdown(email)
