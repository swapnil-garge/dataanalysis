import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Step 1: Data Preparation & Initial Exploration ---
print("### Step 1: Data Preparation & Initial Exploration ###")

# Load your data (replace 'your_machine_data.csv' with your actual file path)
# For demonstration, we'll create a dummy dataframe.
# In your actual use, you would load your CSV like this:
# df = pd.read_csv('your_machine_data.csv')
data = {
    'Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=365)),
    'Uptime': np.random.uniform(85, 99, 365),
    'Yield': np.random.uniform(90, 98, 365),
    'Product_ID': np.random.choice(['Prod_A', 'Prod_B', 'Prod_C', 'Prod_D'], 365),
    'Temperature': np.random.uniform(150, 200, 365),
    'Pressure': np.random.uniform(50, 70, 365),
    'Vibration': np.random.uniform(1, 5, 365),
    'Raw_Material_Batch': np.random.choice(['Batch1', 'Batch2', 'Batch3'], 365)
}
df = pd.DataFrame(data)
# Introduce some uptime drops for demonstration
drop_indices = [30, 90, 150, 210, 270, 330]
for idx in drop_indices:
    df.loc[idx:idx+3, 'Uptime'] -= np.random.uniform(10, 20)
    df.loc[idx, 'Product_ID'] = 'Prod_X' # Simulate a product switch causing a drop


# Data Cleaning & Preparation
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.fillna(method='ffill', inplace=True) # Forward fill missing values

# Feature Engineering: Product Switch
df['Product_Switch'] = (df['Product_ID'].shift(1) != df['Product_ID']).astype(int)

# Initial Visualization: Uptime Plot
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Uptime'], label='Daily Uptime')
plt.title('Machine Uptime Over the Last Year')
plt.xlabel('Date')
plt.ylabel('Uptime (%)')
plt.grid(True)
plt.legend()
plt.show()

# --- Step 2: Identifying Time Trends in Uptime ---
print("\n### Step 2: Identifying Time Trends in Uptime ###")

# Calculate moving averages
df['Uptime_7D_MA'] = df['Uptime'].rolling(window=7).mean()
df['Uptime_30D_MA'] = df['Uptime'].rolling(window=30).mean()

# Plotting the Trends
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Uptime'], label='Daily Uptime', alpha=0.4)
plt.plot(df['Date'], df['Uptime_7D_MA'], label='7-Day Moving Average', linewidth=2)
plt.plot(df['Date'], df['Uptime_30D_MA'], label='30-Day Moving Average', linewidth=2)
plt.title('Uptime Trends with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Uptime (%)')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 3: Pinpointing the Drops ---
print("\n### Step 3: Pinpointing the Drops ###")

# Define a "Drop" using a percentile threshold
uptime_threshold = df['Uptime'].quantile(0.10)
df['Uptime_Drop'] = np.where(df['Uptime'] < uptime_threshold, 1, 0)

# Visualize the Drops
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Uptime'], label='Daily Uptime')
plt.scatter(df[df['Uptime_Drop'] == 1]['Date'], df[df['Uptime_Drop'] == 1]['Uptime'], color='red', label='Significant Uptime Drop')
plt.title('Identified Uptime Drops')
plt.xlabel('Date')
plt.ylabel('Uptime (%)')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 4: Root Cause Analysis - Product Switches & Lags ---
print("\n### Step 4: Root Cause Analysis - Product Switches & Lags ###")

# Product Switch Analysis: Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Product_Switch', y='Uptime', data=df)
plt.title('Uptime on Product Switch vs. Non-Switch Days')
plt.xticks([0, 1], ['No Switch', 'Product Switch'])
plt.ylabel('Uptime (%)')
plt.xlabel('')
plt.show()

# Lag Analysis
for lag in range(1, 6):
    df[f'Product_Switch_Lag_{lag}'] = df['Product_Switch'].shift(lag)

lag_features = [f'Product_Switch_Lag_{lag}' for lag in range(1, 6)]
# Ensure we handle NaN values that are created by shifting
correlation_data = df[lag_features + ['Uptime_Drop']].dropna()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data.corr()[['Uptime_Drop']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Lagged Switches with Uptime Drops')
plt.show()


# --- Step 5: Finding the Best Operating Conditions ---
print("\n### Step 5: Finding the Best Operating Conditions ###")
# List of your process variable columns
process_variables = ['Temperature', 'Pressure', 'Vibration']

# Define High & Low Uptime Categories
high_uptime_threshold = df['Uptime'].quantile(0.90)
df['Uptime_Category'] = 'Normal'
df.loc[df['Uptime'] < uptime_threshold, 'Uptime_Category'] = 'Low'
df.loc[df['Uptime'] > high_uptime_threshold, 'Uptime_Category'] = 'High'

# Compare Process Variable Distributions
print("Comparing Process Variable Distributions across Uptime Categories...")
for var in process_variables:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Uptime_Category', y=var, data=df, order=['Low', 'Normal', 'High'])
    plt.title(f'{var} Distribution Across Uptime Categories')
    plt.show()

# Summarize Optimal Conditions
print("\nCalculating Best Operating Conditions...")
high_uptime_df = df[df['Uptime_Category'] == 'High']
best_operating_conditions = high_uptime_df[process_variables].describe()

print("\n### Best Operating Conditions (Summary from High Uptime Periods) ###")
print(best_operating_conditions)

# Bonus: Feature Importance from a simple model
print("\nBonus: Feature Importance Analysis...")
# One-hot encode categorical features for the model
df_model = pd.get_dummies(df, columns=['Product_ID', 'Raw_Material_Batch'], drop_first=True)
features_for_model = process_variables + [col for col in df_model.columns if 'Product_ID_' in col or 'Raw_Material_Batch_' in col]
X = df_model[features_for_model].copy()
X.fillna(X.mean(), inplace=True) # Handle any remaining NaNs
y = df_model['Uptime']

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get and plot feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances for Predicting Uptime')
plt.xlabel('Importance Score')
plt.show()