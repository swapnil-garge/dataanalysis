import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

# Step 1: Data Collection and Understanding
# Load dataset (replace 'dataset.csv' with actual file path)
df = pd.read_csv('dataset.csv')

# Basic inspection
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Data Heatmap')
plt.show()

# Step 2: Data Preprocessing
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values (impute numerical with median, categorical with mode)
df.fillna(df.select_dtypes(include=np.number).median(), inplace=True)
df.fillna(df.select_dtypes(include='object').mode().iloc[0], inplace=True)

# Create lagged variables for raw material properties (e.g., raw_material_quality)
for lag in [1, 2, 3]:
    df[f'raw_material_quality_lag{lag}'] = df['raw_material_quality'].shift(lag)

# One-hot encode Product
df = pd.get_dummies(df, columns=['Product'], drop_first=True)

# Remove outliers using IQR
Q1 = df['Uptime'].quantile(0.25)
Q3 = df['Uptime'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Uptime'] < (Q1 - 1.5 * IQR)) | (df['Uptime'] > (Q3 + 1.5 * IQR)))]

# Set Date as index
df.set_index('Date', inplace=True)

# Step 3: Identify Time Trends in Uptime
# Plot Uptime over time
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Uptime'], label='Uptime')
plt.title('Uptime Over Time')
plt.xlabel('Date')
plt.ylabel('Uptime (%)')
plt.legend()
plt.show()

# Time-series decomposition
decomposition = seasonal_decompose(df['Uptime'], model='additive', period=30)  # Assuming monthly seasonality
decomposition.plot()
plt.show()

# Identify drops (e.g., below mean - 2*std)
mean_uptime = df['Uptime'].mean()
std_uptime = df['Uptime'].std()
drops = df[df['Uptime'] < (mean_uptime - 2 * std_uptime)]
print("Periods with significant uptime drops:\n", drops[['Uptime']])

# Rolling average
df['Uptime_Rolling'] = df['Uptime'].rolling(window=7).mean()
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Uptime'], label='Uptime')
plt.plot(df.index, df['Uptime_Rolling'], label='7-Day Rolling Avg', color='red')
plt.title('Uptime with Rolling Average')
plt.xlabel('Date')
plt.ylabel('Uptime (%)')
plt.legend()
plt.show()

# Step 4: Investigate Causes of Uptime Drops
# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Random Forest for feature importance
X = df.drop(['Uptime', 'Uptime_Rolling'], axis=1, errors='ignore')
y = df['Uptime']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importances:\n", importances.sort_values(ascending=False))

# Granger causality test (example for raw_material_quality_lag1)
granger_data = df[['Uptime', 'raw_material_quality_lag1']].dropna()
granger_results = grangercausalitytests(granger_data, maxlag=3, verbose=True)

# Step 5: Analyze Product Switches
# Create product switch indicator
df['Product_Switch'] = (df['Product'].shift(1) != df['Product']).astype(int)

# Compare uptime during switch vs. non-switch
switch_uptime = df[df['Product_Switch'] == 1]['Uptime']
no_switch_uptime = df[df['Product_Switch'] == 0]['Uptime']
t_stat, p_value = ttest_ind(switch_uptime, no_switch_uptime, equal_var=False)
print(f"T-test for Uptime (Switch vs. No-Switch): p-value = {p_value}")

# Uptime by product
plt.figure(figsize=(12, 6))
sns.boxplot(x='Product', y='Uptime', data=df)
plt.title('Uptime by Product')
plt.xticks(rotation=45)
plt.show()

# Step 6: Identify Best Operating Conditions
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
print("Model RMSE:", np.sqrt(mean_squared_error(y_test, rf.predict(X_test))))

# Partial dependence plots for top features
features = importances.sort_values(ascending=False).index[:3]
PartialDependenceDisplay.from_estimator(rf, X_train, features)
plt.show()

# Analyze high-uptime periods
high_uptime = df[df['Uptime'] > df['Uptime'].quantile(0.9)]
print("Summary of variables during high uptime:\n", high_uptime.describe())