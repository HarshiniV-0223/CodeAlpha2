import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.express as px

# ------------------ 1. Load and Clean Data ------------------
data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data.drop_duplicates(inplace=True)

if 'Region.1' in data.columns:
    data.drop(columns=['Region.1'], inplace=True)

# ------------------ 2. Basic Exploration ------------------
print("\nTotal rows and columns:", data.shape)
print("\n--- Summary Statistics ---")
print(data.describe())
print("\nUnique Regions:", data['Region'].nunique())
print("Regions List:", data['Region'].unique())

# ------------------ 3. Data Preparation ------------------
regionwise = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values()
covid_period = data[(data['Date'] >= '2020-03-01') & (data['Date'] <= '2020-12-31')]
data['Month'] = data['Date'].dt.month
heatmap_data = data.pivot_table(values='Estimated Unemployment Rate (%)',
                                index='Region', columns='Month', aggfunc='mean')

# ------------------ 4. Dashboard Layout ------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
plt.subplots_adjust(wspace=0.4)

# Graph 1: Trend Over Time
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)',
             data=data, color='blue', marker='o', ax=axes[0])
axes[0].set_title("📈 Trend Over Time", fontsize=14)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Rate (%)")
axes[0].tick_params(axis='x', rotation=45)

# Graph 2: Average Unemployment by Region
regionwise.plot(kind='bar', color='green', width=0.6, ax=axes[1])
axes[1].set_title("🏙 Average Unemployment by Region", fontsize=14)
axes[1].set_ylabel("Rate (%)")

# Graph 3: Covid-19 Impact
sns.scatterplot(x='Date', y='Estimated Unemployment Rate (%)',
                data=covid_period, color='red', ax=axes[2])
axes[2].set_title("🦠 Covid-19 Impact (2020)", fontsize=14)
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Rate (%)")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ------------------ 5. Large Heatmap ------------------
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f")
plt.title("📊 Seasonal Monthly Unemployment Rate Heatmap", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# ------------------ 6. Prediction Model ------------------
data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
X = data[['Date_ordinal']]
y = data['Estimated Unemployment Rate (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

future_dates = pd.date_range(start=data['Date'].max(), periods=7, freq='M')
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
future_preds = model.predict(future_ordinal)

plt.figure(figsize=(10, 5))
plt.plot(data['Date'], y, label='Historical Data', color='blue')
plt.plot(future_dates, future_preds, label='Predicted', color='orange', marker='o')
plt.title("🔮 Unemployment Rate Forecast")
plt.xlabel("Date")
plt.ylabel("Rate (%)")
plt.legend()
plt.tight_layout()
plt.show()

map_data = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.bar(map_data['Region'], map_data['Estimated Unemployment Rate (%)'], color='purple')
plt.xticks(rotation=90)
plt.title("🗺 Unemployment Rate Across India (by Region)")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()
