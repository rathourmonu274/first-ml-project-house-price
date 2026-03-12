
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.model_selection import train_test_split  # tyoe: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from sklearn.datasets import fetch_california_housing  # type: ignore
from sklearn.model_selection import train_test_split   # type: ignore
from sklearn.linear_model import LinearRegression      # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore


# Step 2: Load the dataset
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("First 5 rows of dataset:")
print(df.head())


# Step 3: EDA (Exploratory Data Analysis)
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# Step 4: Define features and target
X = df.drop('target', axis=1)
y = df['target']


# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Model Selection
model = LinearRegression()


# Step 7: Model Training
model.fit(X_train, y_train)


# Step 8: Prediction
y_pred = model.predict(X_test)


# Step 9: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Step 10: Actual vs Predicted Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


# ==========================================
# step 1: load the data and inspect the data (for null value)
# ==========================================
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\iot\House_Price_Data.csv")

# Display first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for null values
print("\nNull Value Check:")
print(df.isnull().sum())

# Drop nulls if any (Standard practice)
df.dropna(inplace=True)

# ==========================================
# step 2: define features (x) and target (y)
# ==========================================

x = df[['Square_Feet']]
y = df['Price']

# ==========================================
# step 3: train test
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

# ==========================================
# step 4: prediction and evaluation
# ==========================================
y_pred = lr.predict(X_test)

# Calculate Accuracy (R2 Score) and Error (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score (Accuracy): {r2}")
print(f"Mean Squared Error: {mse}")

# Show formula (y = mx + c)
m = lr.coef_[0]
c = lr.intercept_
print(f"Formula: Price = {m:.2f} * Square_Feet + {c:.2f}")

# ==========================================
# step 5: visual of the result
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the actual data points (Scatter plot)
sns.scatterplot(x='Square_Feet', y='Price', data=df,
                color='blue', label='Actual Data', s=100)

# Plot the regression line (The prediction line)
plt.plot(X_test, y_pred, color='red', linewidth=3,
         label='Regression Line (Prediction)')

plt.title('House Price Prediction (Linear Regression)')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# after completion
print("done plotting")

