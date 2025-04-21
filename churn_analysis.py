# churn_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/telco_churn.csv')

# Basic info
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.show()

# Correlation heatmap for numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
