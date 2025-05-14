# Fraud Detection System for Financial Transactions

A machine learning system to detect fraudulent credit card transactions using real-world, highly imbalanced data.

## 📝 Description

This project uses the **Kaggle Credit Card Fraud Detection** dataset, applying various preprocessing, feature engineering, and modeling techniques to predict fraudulent activity.

## 📊 Dataset

- Source: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Highly imbalanced (0.17% fraud)

## 📁 Project Structure

```
├── data/               # Raw dataset
├── notebooks/          # EDA & experiments
├── src/                # Source scripts
├── main.py             # Pipeline script
├── requirements.txt    # Dependencies
```

## 🔧 Techniques Used

- SMOTE for balancing
- Random Forest, XGBoost
- Precision, Recall, AUC
- ROC curve, Confusion Matrix

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Place dataset in data/
# Step 3: Run the pipeline
python main.py
```

## 📈 Results

- High Recall (reduced false negatives)
- Precision-Recall and ROC curves plotted
- Confusion matrix for evaluation

## 🤝 Contributing

Feel free to fork and contribute!
