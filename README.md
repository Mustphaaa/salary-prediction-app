# 💰 Salary Prediction App

An end-to-end machine learning project: a Linear Regression model that predicts salary from years of experience, served through an interactive Streamlit web app.

**Live app:** _add your Streamlit Community Cloud URL here_

## Project description
The model learns the relationship between years of experience and salary. The Streamlit app loads the saved model (`model/model.pkl`), takes a user's years of experience, and displays the predicted salary along with a chart, dataset information, model metrics, and prediction history.

## Dataset
Salary Dataset (`Salary_Data.csv`) – 40 rows, 2 columns: `Experience Years`, `Salary`. Provided in class.

## Model
- Algorithm: Simple Linear Regression (scikit-learn)
- Feature: `Experience Years` | Target: `Salary`
- Split: 80% train / 20% test (`random_state=42`)
- Serialized with `joblib` to `model/model.pkl`

## Evaluation metrics (test set)
| Metric | Value |
|--------|-------|
| R²     | 0.907 |
| MAE    | 6,420 |
| RMSE   | 6,934 |
| MSE    | 48,077,731 |

## Project structure
```
ML_Project/
├── data/Salary_Data.csv
├── model/model.pkl
├── model/metrics.json
├── notebooks/model_training.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally
```bash
git clone <your-repo-url>
cd ML_Project
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
Open `notebooks/model_training.ipynb` and run all cells. It overwrites `model/model.pkl` and `model/metrics.json`.

## Tech stack
Python, pandas, scikit-learn, matplotlib, Streamlit, joblib.
