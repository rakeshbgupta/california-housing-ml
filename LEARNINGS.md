# What I Learned From This Project 🚀

This project represents my first complete end-to-end Machine Learning workflow implemented in Python using VS Code and GitHub.

Below is a summary of what I learned and practiced during this project.

---

## 1. Core Machine Learning Workflow

- Loaded a real-world public dataset (California Housing) using `scikit-learn`
- Understood the difference between **features (X)** and **target (y)**
- Performed a **train/test split** with proper **stratification** for classification
- Built a **baseline classification model** using Logistic Regression
- Evaluated the model using **accuracy** and interpreted results meaningfully

---

## 2. Exploratory Data Analysis (EDA)

- Used **pandas** to inspect data (`info()`, `describe()`, missing values)
- Computed and interpreted **correlations**
- Created **scatter plots** using `matplotlib`
- Learned how the **alpha parameter** helps visualise dense data points
- Observed real geographic and economic patterns in housing prices

---

## 3. Feature Understanding & Interpretation

- Understood why `MedInc` (median income) is the strongest predictor
- Learned how geographic features (latitude, longitude) influence prices
- Interpreted **model coefficients** to understand feature impact
- Connected statistical output to real-world intuition

---

## 4. Python Project Structure (Very Important)

- Structured a Python ML project using:
  - `src/` for reusable code
  - `notebooks/` for EDA
  - `main.py` as the pipeline entry point
- Learned why and how to use:
  ```python
  if __name__ == "__main__":
      main()


# 5. Writing Clean, Professional Python Code

Used functions instead of writing everything inline

Learned when and why to:

split code across files

Used type hints like Tuple[pd.DataFrame, ...] for readability

Understood the difference between:
runtime types (tuple)
type hints (typing.Tuple)

# 6. Logging & Debugging

Used Python’s logging module instead of print

Learned why %s is used in logging (lazy formatting)

Logged:

dataset shapes

sample rows

pipeline progress

Understood why logging is important for ML pipelines

# 7. Defensive Programming & Validation

Learned how to:

validate assumptions between pipeline steps

raise custom exceptions when data mismatches occur

Understood why failing fast is critical in data/ML pipelines

# 8. Model Training Details

Learned what max_iter means in Logistic Regression

Recognised the importance of convergence warnings

Understood the role of feature scaling (conceptually)

# 9. Version Control & GitHub

Initialised a local git repository

Committed meaningful changes with descriptive messages

Created a public GitHub repository

Pushed a complete, working ML project to GitHub

Understood how this forms part of an ML/AI portfolio
