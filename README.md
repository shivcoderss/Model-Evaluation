# 🔍 Machine Learning Model Comparison & Hyperparameter Tuning

This Streamlit web application allows you to train and compare multiple machine learning classifiers with built-in hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`. It simplifies the process of evaluating models across several performance metrics such as **accuracy**, **precision**, **recall**, and **F1-score**, and enables you to select the best model based on actual results.

### 🚀 Features

- Train Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM)
- Interactive selection of models from the sidebar
- Choose between Grid Search or Randomized Search for hyperparameter tuning
- Automatic scaling and preprocessing
- Compare results in a structured, sortable DataFrame
- Lightweight and deployable via [Streamlit Cloud](https://streamlit.io/cloud)

### 📦 Tech Stack

- Python
- Scikit-learn
- Streamlit
- NumPy
- Pandas

### 📂 Project Structure

- ml-model-comparison/
- ├── app.py # Main Streamlit app
- ├── requirements.txt # Required Python packages
- └── README.md # App documentation
  
### ▶️ Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
