# California Housing – EDA + Simple ML Model

Small end-to-end machine learning project to practise the basic ML workflow:

- Load the California housing dataset
- Exploratory Data Analysis (EDA) in a Jupyter notebook
- Create a binary target (`HighValue`) based on median house value
- Train a Logistic Regression classifier
- Evaluate accuracy and inspect feature coefficients

## Project Structure

```text
california-housing-ml/
├── notebooks/
│   └── california_housing_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── split.py
│   └── model_train.py
├── main.py
├── requirements.txt
└── .gitignore
