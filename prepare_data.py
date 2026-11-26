import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/real_estate.csv")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("5 dong dau:")
print(data.head())
print("\nKich thuoc:", data.shape)
print("\nThong tin cac cot:")
data.info()
print("\nThống kê mô tả:")
print(data.describe())

data = data.drop(columns=["No"])
X = data.drop(columns=["Y house price of unit area"])
Y = data["Y house price of unit area"]

print("\nX:", X.shape)
print(X.head())
print("\nY:", Y.shape)
print(Y.head())

Y.hist(bins=30)
plt.title("Phan bo gia nha")
plt.xlabel("Gia nha")
plt.ylabel("So luong")
plt.show()