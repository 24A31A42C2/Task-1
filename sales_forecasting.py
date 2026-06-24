import pandas as pd

df = pd.read_csv("stores_sales_forecasting.csv", encoding="latin1")

print(df.head())
print(df.shape)
print(df.columns)

print(df[['Order Date', 'Sales']].head())
df['Order Date'] = pd.to_datetime(df['Order Date'])

print("Start Date:", df['Order Date'].min())
print("End Date:", df['Order Date'].max())
daily_sales = df.groupby('Order Date')['Sales'].sum()

print(daily_sales.head())
print(daily_sales.isnull().sum())
print("Average Daily Sales:", daily_sales.mean())
import matplotlib.pyplot as plt

daily_sales.plot(figsize=(10,5))

plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
print("Average Daily Sales:", daily_sales.mean())
from sklearn.linear_model import LinearRegression
import numpy as np

# Create day numbers
daily_sales_df = daily_sales.reset_index()
daily_sales_df['Day'] = np.arange(len(daily_sales_df))

# Features and target
X = daily_sales_df[['Day']]
y = daily_sales_df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_days = np.arange(len(daily_sales_df), len(daily_sales_df) + 30).reshape(-1, 1)
future_sales = model.predict(future_days)

print("Next 30 Days Sales Forecast:")
print(future_sales)
plt.figure(figsize=(10,5))
plt.plot(daily_sales_df['Day'], y, label='Historical Sales')
plt.plot(future_days, future_sales, label='Forecasted Sales')
plt.title("Sales Forecast for Next 30 Days")
plt.xlabel("Day Number")
plt.ylabel("Sales")
plt.legend()
plt.show()