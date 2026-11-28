import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/real_estate.csv")
df = df.drop(columns=["No"])
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("5 dong dau:")
print(df.head())
print("\nKich thuoc:", df.shape)
print("\nThong tin cac cot:")
df.info()
print("\nThong ke mo ta:")
print(df.describe())
data = df.to_numpy()
X = data[:, :-1]
y = data[:, -1]
print("\nX:", X.shape)
print(X[:5])
print("\ny:", y.shape)
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
plt.hist(y, bins=30)
plt.title("Phan bo gia nha")
plt.xlabel("Gia nha")
plt.ylabel("So luong")
plt.show()
