import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("🔍 Machine Learning Model Comparison & Hyperparameter Tuning")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10]
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

selected_models = st.sidebar.multiselect("Select models to train", list(models.keys()), default=list(models.keys()))
search_method = st.sidebar.radio("Search Method", ['GridSearchCV', 'RandomizedSearchCV'])

if st.sidebar.button("Run Tuning & Evaluation"):
    with st.spinner("Training models..."):
        results = []

        for name in selected_models:
            model = models[name]
            params = param_grids[name]

            st.write(f"🔧 Tuning **{name}**...")
            if search_method == "GridSearchCV" or name in ['Logistic Regression', 'Decision Tree']:
                search = GridSearchCV(model, params, cv=5)
            else:
                search = RandomizedSearchCV(model, params, n_iter=5, cv=5, random_state=42)

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            scores = evaluate_model(name, best_model)
            scores["Best Params"] = str(search.best_params_)
            results.append(scores)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="F1 Score", ascending=False)

        st.subheader("📊 Model Performance")
        st.dataframe(results_df, use_container_width=True)

        st.success("Done!")
else:
    st.info("👈 Select models and click 'Run Tuning & Evaluation'")
