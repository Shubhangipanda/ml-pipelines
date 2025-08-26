# ML Pipelines

End-to-end machine learning workflow (EDA → training → deployment).

Some notes:
1) Preprocessing:
One-Hot Encoding (pd.get_dummies)

Computers don’t understand words like "Month-to-month" or "Two year".
We turn them into numbers using 0s and 1s (like checkboxes).

Example with the Contract column:

Before: ["Month-to-month", "One year", "Two year"]

After:

Contract_One year
Contract_Two year

If both are 0 → it means "Month-to-month".
drop_first=True removes the first column to save space (no duplicate info).

*Why it helps later (for ML models)

Machine learning models (like Logistic Regression, RandomForest, XGBoost) can only work with numbers.

They can’t directly understand text like "One year" or "Two year".

If you leave text as-is, the model will throw an error or treat them all as the same thing (which is wrong).

2) ## 📐 Math Behind the Project (Simplified)

This project predicts **Customer Churn** (whether a telecom customer will stay or leave).  
Here’s the simple math explained:

---

### 🔹 1. Data Preprocessing
- **Scaling (Standardization):**  
  We transform numbers so they are on the same scale:  
  \[
  z = \frac{x - \mu}{\sigma}
  \]  
  (mean = 0, std = 1).  

- **Encoding Categories:**  
  Words like `"Month-to-Month"` or `"Two Year"` are converted into numbers using **dummy variables (0/1 columns)**.

---

### 🔹 2. Logistic Regression (The Model)
- **Linear Combination:**  
  Combine features with weights:  
  \[
  z = w_1 x_1 + w_2 x_2 + ... + b
  \]  

- **Sigmoid Function:**  
  Converts result into probability between 0 and 1:  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]  
  → Probability of churn.  

- **Loss Function (Binary Cross-Entropy):**  
  Punishes wrong predictions, especially confident wrong ones:  
  \[
  L = -\frac{1}{m} \sum \big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \big]
  \]  

- **Gradient Descent:**  
  Updates weights step by step to reduce the loss.

---

### 🔹 3. Evaluation Metrics
- **Confusion Matrix:**  
  - TP = predicted churn & actually churned  
  - TN = predicted stay & actually stayed  
  - FP = predicted churn but they stayed  
  - FN = predicted stay but they churned  

- **Accuracy:** overall correctness  
- **Precision:** of predicted churners, how many really churned  
- **Recall:** of all churners, how many we caught  
- **F1 Score:** balance between precision and recall  

---

✅ In short:  
We clean data → scale & encode → train logistic regression → predict churn probability → evaluate with metrics.

3)
## 📊 Random Forest Model Results  

We trained a **Random Forest Classifier** to predict customer churn.  

### 🔑 Key Metrics
| Metric       | Score |
|--------------|-------|
| Accuracy     | **0.78** |
| Precision    | **0.62** |
| Recall       | **0.48** |
| F1 Score     | **0.54** |
| ROC-AUC      | *(to be added after calculation)* |

---

### ✅ What These Metrics Mean
- **Accuracy (0.78)** → The model predicts churn correctly about 78% of the time.  
- **Precision (0.62)** → Out of customers predicted as churners, 62% actually churned.  
- **Recall (0.48)** → The model caught 48% of the actual churners.  
- **F1 Score (0.54)** → Balance between precision and recall.  
- **ROC-AUC** → Will tell us how well the model separates churn vs no churn.  

---

### 📌 Insights from Random Forest
- **Non-churners (0 class)** are predicted much better than churners (1 class).  
- Customers with:  
  - **Short tenure**  
  - **High monthly charges**  
  - **Month-to-month contracts**  
  are more likely to churn.  

---

### 💾 Model Artifact
The trained Random Forest model is saved at:
```
models/churn_rf_model.pkl
```

### 🔹 What is ROC?
- ROC = *Receiver Operating Characteristic*  
- It’s a curve that shows how well the model separates:  
  - **Churners (1)** vs **Non-Churners (0)**  
- X-axis → False Positive Rate (wrongly saying someone will churn)  
- Y-axis → True Positive Rate (catching actual churners)  

### 🔹 What is AUC?
- AUC = *Area Under the ROC Curve*  
- Score between **0 and 1**  
  - **1.0** → Perfect model  
  - **0.5** → Random guessing  
  - **>0.8** → Very good model  

### 🔹 Why it matters?
- Accuracy can be misleading if classes are imbalanced.  
- ROC-AUC tells us how well the model **truly separates churners vs non-churners**.  

