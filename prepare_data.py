import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/real_estate.csv")
df = df.drop(columns=["No"])
data = df.to_numpy()
X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(data.shape, X.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)