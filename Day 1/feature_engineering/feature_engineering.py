import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

file_name = "sales_data.csv"

if not os.path.exists(file_name):
    print(f"Error: {file_name} is not found")
    exit()


df = pd.read_csv(file_name)
print("Sucessfully loaded")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print(df.isnull().sum())
print()


df["Age"] = df["Age"].fillna(df["Age"].median())
print(df.isnull().sum())
print()

df["Spending"] = df["Spending"].fillna(df["Spending"].median())
print(df.isnull().sum())
print()

df["Visits_Per_Month"] = df["Visits_Per_Month"].fillna(df["Visits_Per_Month"].median())
print(df.isnull().sum())
print()

# plt.figure(figsize=(4,2))
# df["Spending"].hist(bins=10, color="green", edgecolor="black")
# plt.title("Distribution of Spending")
# plt.xlabel("Spending Anount")
# plt.ylabel("Number of Customers")
# plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x=df["Age"], color="skyblue")
plt.title("Boxplot of Customer Age")
plt.xlabel("Age")
plt.show()