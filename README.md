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
