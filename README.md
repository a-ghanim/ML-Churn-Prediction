# Customer Churn Prediction Application

This repository hosts a **Customer Churn Prediction** application, which uses a variety of pre-trained machine learning models to assess the likelihood of customer churn. Built with **Streamlit**, the application enables users to input customer data, view churn probabilities, receive model-based explanations, and generate personalized retention emails for high-risk customers. The application is accessible on [Hugging Face Spaces](https://huggingface.co/spaces/a-ghanim/ml-customer-churn-prediction).

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Notebook](#notebook)
5. [Input Features](#input-features)
6. [How It Works](#how-it-works)
7. [Adding New Models](#adding-new-models)
8. [Dependencies](#dependencies)
9. [Using an Alternative LLM](#using-an-alternative-llm)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

This project utilizes multiple machine learning models, including Decision Tree, KNN, Naive Bayes, Random Forest, SVM, XGBoost, and an ensemble Voting Classifier, to predict customer churn probability. Through a user-friendly web interface, it provides insights into customer behavior and offers tools for customer retention.

---

## Features

- **Churn Prediction**: Calculates churn probability using various machine learning models for reliable insights.
- **Interpretive Explanations**: Highlights key factors driving predictions, adding transparency to churn risk.
- **Visual Analytics**: Displays churn probabilities for each model and overall risk with intuitive charts.
- **Customer Engagement Drafts**: Generates personalized retention emails for at-risk customers.

---

## Getting Started

To set up and run this project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

### 2. Install Dependencies

Ensure Python 3.7+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up the Groq API Key

Add your **Groq API key** as an environment variable:

- **For local setups**: create a `.env` file in the project root:
   ```plaintext
   GROQ_API_KEY=your_api_key_here
   ```
   Install `python-dotenv` to load the API key:
   ```bash
   pip install python-dotenv
   ```

- **For cloud platforms** (e.g., Replit, Hugging Face Spaces): add `GROQ_API_KEY` directly in the platform’s "secrets" or environment variable settings.

In `app.py`, load the API key:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   api_key = os.getenv("GROQ_API_KEY")
   ```

### 4. Start the Streamlit App

Run the app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Notebook

The `ml-customer-churn-prediction.ipynb` notebook covers data preparation, model training, and evaluation:

1. **Data Exploration**: Loads and examines the `churn.csv` dataset, analyzing distribution and trends.
2. **Exploratory Data Analysis**: Visualizes churn patterns by demographic and financial attributes.
3. **Feature Engineering**: Processes and encodes features, adding variables like `CLV` and `TenureAgeRatio`.
4. **Model Training & Evaluation**: Trains models (Logistic Regression, XGBoost, etc.), evaluates accuracy, and saves models as `.pkl` files.
5. **Feature Importance**: Uses XGBoost to identify features that significantly affect churn predictions.
6. **Advanced Techniques**: Uses SMOTE to handle class imbalance and combines models with a Voting Classifier.

---

## Input Features

The model calculates churn probability based on these customer attributes:

- **Credit Score**
- **Geography** (Country: France, Germany, or Spain)
- **Gender**
- **Age**
- **Tenure** (Years with the bank)
- **Balance**
- **Number of Products**
- **Has Credit Card** (Yes/No)
- **Is Active Member** (Yes/No)
- **Estimated Salary**

These features form the basis of each model’s churn prediction.

---

## How It Works

1. **Prediction**: Pre-trained models calculate churn probability based on input features, displaying individual model probabilities.
2. **Explanation**: The app uses the `Groq` API and `gemma2-9b-it` model to generate insights into the key features driving predictions.
3. **Engagement Email**: Uses `Groq` to generate a personalized email draft for retention strategies.

---

## Adding New Models

To add a new model:

1. **Train and Save the Model**:
   ```python
   import pickle
   with open("new_model.pkl", "wb") as file:
       pickle.dump(new_model, file)
   ```

2. **Add the Model File**: Place the `.pkl` file in the project directory or a `models` folder.

3. **Load the Model in `app.py`**:
   ```python
   def load_model(filename):
       with open(filename, "rb") as file:
           return pickle.load(file)
   new_model = load_model('new_model.pkl')
   ```

4. **Update `make_predictions`**:
   ```python
   def make_predictions(input_df):
       probabilities = {
           'XGBoost': float(xgboost_model.predict_proba(input_df)[0][1]),
           'RandomForest': float(random_forest_model.predict_proba(input_df)[0][1]),
           'NewModel': float(new_model.predict_proba(input_df)[0][1])
       }
       avg_probability = np.mean(list(probabilities.values()))
       return avg_probability
   ```

5. **Test the Integration**: Run the app and verify the new model’s predictions.

---

## Dependencies

```plaintext
numpy
pandas
plotly
streamlit
xgboost
scikit-learn
python-dotenv
```

- **scikit-learn**: Required for models like Decision Tree, KNN, Naive Bayes, Random Forest, SVM, and Voting Classifier.
- **xgboost**: For XGBoost models.
- **streamlit** and **plotly**: For the app interface and visualizations.
- **python-dotenv**: Loads the Groq API key from a `.env` file.

---

## Using an Alternative LLM

The current setup uses **Groq's `gemma2-9b-it` model** for explanations and email generation. If you wish to substitute it with another LLM (e.g., OpenAI’s GPT-3, Anthropic’s Claude), adjust the `groq_completion` function to use the desired API.

Simply replace the `client.chat.completions.create()` call with the equivalent method in the chosen LLM's API. Be sure to update any specific configuration or environment variables required by the new model.

---

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request. Include documentation and tests where necessary.

---

## License

This project is licensed under the MIT License.

---
