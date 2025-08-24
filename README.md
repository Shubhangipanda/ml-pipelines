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

