# 💳 Credit Risk Scoring System — From Data to Decision (PD, LGD, EAD, Expected Loss)

## 📌 Introduction

In financial institutions, one of the most critical challenges is to assess the risk associated with lending money to clients. This project proposes a complete and practical solution to this problem by building a **credit risk scoring system** grounded in real-world methodologies used in banking and risk management.

The objective is not only to predict whether a client will default, but to **quantify the financial impact of that default** and provide **interpretable, actionable insights**.

This is achieved through the estimation of three key components:

- **PD (Probability of Default)**: likelihood that a borrower will default  
- **LGD (Loss Given Default)**: proportion of loss if default occurs  
- **EAD (Exposure at Default)**: total exposure at the time of default  

These components are combined to compute the most important risk metric:

> **Expected Loss (EL) = PD × LGD × EAD**

---

## 🎯 Problem Statement

The core problem is formulated as a **supervised learning task**:

Given borrower characteristics \( X \), predict:

- \( Y = 1 \) if the borrower defaults  
- \( Y = 0 \) otherwise  

The goal is to estimate:

> **PD = P(Y = 1 | X)**

Beyond classification, the project extends this prediction into a **financial risk estimation framework**, which is closer to real-world industry practices.

---

## 📊 Data Sources

The models are trained using well-known public datasets:

- **German Credit Dataset**
- **Home Credit Default Risk (Kaggle)**

These datasets include a rich set of variables describing borrower profiles:

- Financial variables (income, loan amount, interest rate)
- Behavioral indicators (credit usage, delinquencies)
- Demographic information (employment, housing)

---

## ⚠️ Important: Avoiding Data Leakage

A critical step in credit modeling is ensuring that the model does not use information that would not be available at decision time.

For example, variables such as:

- loan_status  
- recoveries  
- total payments  

are removed because they contain **future information**.

This ensures that the model remains **realistic and deployable**.

---

## ⚙️ Methodology

### 1. Data Preprocessing

The raw data is first transformed into a clean and usable format:

- Handling missing values (imputation strategies)
- Encoding categorical variables (one-hot encoding)
- Scaling numerical features
- Splitting into training and testing sets

This step ensures that the models can learn effectively without bias or noise.

---

### 2. Feature Engineering

Feature engineering plays a key role in improving model performance.

New variables are created to better capture risk:

- Debt-to-Income ratio (financial pressure indicator)
- Employment length (stability proxy)
- Credit behavior indicators

These transformations allow the model to better reflect **economic intuition**.

---

### 3. Modeling Strategy

Several models are evaluated to compare performance and robustness:

- Logistic Regression (baseline, interpretable)
- Random Forest (non-linear relationships)
- XGBoost / LightGBM (state-of-the-art performance)
- Support Vector Machine (margin-based learning)

This multi-model approach ensures that we identify the best trade-off between **performance and interpretability**.

---

### 4. Model Evaluation

The models are evaluated using metrics aligned with risk management objectives:

- **ROC-AUC**: ability to rank risky clients correctly  
- **F1-score**: balance between precision and recall  
- **Precision / Recall**: important for risk decisions  
- **Confusion Matrix**: detailed error analysis  

The focus is not only accuracy, but also **business impact**.

---

## 🔍 Model Explainability (SHAP)

In finance, predictions must be explainable.

This project integrates **SHAP (SHapley Additive Explanations)** to:

- Identify the most important features globally  
- Explain individual predictions (client-level decisions)  
- Provide transparency for regulatory and business needs  

This transforms the model from a "black box" into a **decision-support tool**.

---

## 💰 Credit Risk Modeling

The true value of this project lies in connecting machine learning predictions to financial risk.

Each client is evaluated using:

- **PD** → predicted by the model  
- **LGD** → estimated or assumed (e.g., 60%)  
- **EAD** → loan exposure  

These are combined into:

> **Expected Loss = PD × LGD × EAD**

This metric allows institutions to:

- Quantify risk in monetary terms  
- Prioritize high-risk clients  
- Optimize lending strategies  

---

## 🚀 Deployment

To make the system usable in practice, the project includes deployment components:

### API (FastAPI)
- Real-time prediction service  
- Accepts client data as input  
- Returns PD, LGD, EAD, and Expected Loss  

### Web Application (Streamlit)
- Interactive dashboard  
- Allows non-technical users to assess risk  
- Visualizes predictions and explanations  

This ensures the project is not just theoretical, but **operational**.

---

## 📁 Project Structure

The project is organized to reflect professional standards:

