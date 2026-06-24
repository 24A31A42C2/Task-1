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
