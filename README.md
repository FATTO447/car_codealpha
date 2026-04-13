# 🚗 Car Decision Helper
### From a Simple Prediction Task → A Market Intelligence Tool

---
## 💡 The Problem

Every day, people buy used cars without really knowing:
- Is this price fair?
- Does the mileage make sense for this car's age?
- Should I buy from a dealer or an individual?
- Is this car losing value faster than average?

**They rely on gut feeling. I built data to replace that.**

---

## 🎯 What I Built

The task was simple: *"Train a regression model to predict car prices."*

But I asked a different question:
> **"If this were a real business tool — what would it actually need to do?"**

So instead of just a model, I built a **3-page decision support system:**

| Page | Purpose |
|------|---------|
| 🔍 Factor Analysis | What actually drives car prices? |
| 🚗 Car Explorer | Deep dive into any car in the market |
| 🎯 Car Decision Helper | Get a price prediction + buy/skip verdict |

---

## 📊 The Data Challenge

- **299 cars only** — a small dataset by ML standards
- Indian used car market (2003–2018)
- Despite the size, extracted meaningful insights through **feature engineering**

---

## ⚙️ Feature Engineering

| Feature | Logic |
|---------|-------|
| `Car_Age` | 2025 - Year |
| `Depreciation_Rate` | (Present - Selling) / Present × 100 |
| `km_Per_year` | KMs / Car Age |
| `Is_First_Owner` | Binary — ownership history |
| `Is_Dealer` | Binary — seller type |
| `KM_Category` | Low / Medium / High |

---

## 🤖 Models Compared

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 2.549 | 4.447 | 0.233 |
| Random Forest | 2.246 | 3.907 | 0.408 |
| **XGBoost** ✅ | **2.175** | **3.772** | **0.506** |

> R² = 0.50 is the ceiling for this dataset size.
> The value isn't in the number — it's in **what the model tells us.**

---

## 🔍 Key Insights Discovered

- **Is_Dealer** is the #1 price driver (29% importance) — not fuel or age
- Diesel cars hold value better despite higher mileage
- First owner cars sell for 2.7L more on average
- Cars lose the most value in their first 3 years

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `Scikit-learn` · `Pandas` · `Plotly` · `Streamlit`

---

## 🚀 Future Vision

If this were a real product:
- Live data from car listing platforms (Hatla2ee, Syarah)
- Arabic market support (Egypt, Saudi, UAE)
- Price negotiation advisor
- Maintenance cost predictor

---

## 📁 Project Structure

```
car_decision_helper/
├── notebook.ipynb          ← Full analysis & model training
├── app.py                  ← Streamlit web app
├── cleaned_car_data_v2.csv ← Processed dataset
├── xgb_car_price_model.pkl ← Trained XGBoost model
└── README.md
```
