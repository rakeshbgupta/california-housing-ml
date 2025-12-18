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


## What I Learned From This Project

- Implemented a complete end-to-end Machine Learning workflow in Python using VS Code and GitHub
- Performed exploratory data analysis (EDA) using pandas and matplotlib to understand feature distributions and correlations
- Built a baseline classification model using Logistic Regression and evaluated it using accuracy (~83%)
- Understood the importance of proper train/test splitting and class stratification for reliable evaluation
- Interpreted model coefficients to understand feature importance and real-world impact
- Structured a Python ML project using reusable modules (`src/`), notebooks for EDA, and a clean entry point (`main.py`)
- Gained practical understanding of Python concepts such as type hints, logging, custom exceptions, and the `__name__ == "__main__"` pattern
- Learned how to version, document, and publish a working ML project to GitHub as part of a professional portfolio
