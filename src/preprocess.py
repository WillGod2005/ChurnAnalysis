import pandas as pd
import train_test_split
import StandardScaler
from sklearn import labelEncoder

def preprocess_data(path='../data/telco_churn.csv'):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    x = df.drop('churn', axis=1)
    y = df['churn']

    x = pd.get_dummies(x)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
