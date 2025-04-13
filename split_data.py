import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys

def data_reader(file):
    column_names = ['id', 'diagnosis']
    for i in range(30):  
        column_names.append(f'feature_{i}')

    df = pd.read_csv(file, header=None, names=column_names)
    df = df.drop('id', axis=1)
    return df

def preprocessing(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    labelencoder = LabelEncoder()
    standardscaler = StandardScaler()
    y = 1 - labelencoder.fit_transform(y)
    X_scaled = standardscaler.fit_transform(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid

def main():
    try :
        df = data_reader(sys.argv[1])
        X_train, X_valid, y_train, y_valid = preprocessing(df)
        np.save("X_train.npy", X_train)
        np.save("X_valid.npy", X_valid)
        np.save("y_train.npy", y_train)
        np.save("y_valid.npy", y_valid)
    except FileNotFoundError:
        print(f'{sys.argv[1]} not found')

if __name__ == "__main__":
    main()