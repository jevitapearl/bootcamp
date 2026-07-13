import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

# print("Understanding dataset")

file_name = "sales_data.csv"

if not os.path.exists(file_name):
    print(f"Error: {file_name} is not found")
    exit()


df = pd.read_csv(file_name)
# print("Sucessfully loaded")
# print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# print()
# print(df.head())
# print(df.tail())
# print(df.describe())

# median_age = df["Age"].median()
# df["Age"] = df["Age"].fillna(median_age)
# print(median_age)

# print()
# print(df.isnull().sum())
# print()

# median_spending = df["Spending"].median()
# df["Spending"] = df["Spending"].fillna(median_spending)
# print(median_spending)

# print()
# print(df.isnull().sum())

# plt.figure(figsize=(4,2))
# df["Spending"].hist(bins=10, color="skyblue", edgecolor="black")
# plt.title("Distribution of Spending")
# plt.xlabel("Spending Anount")
# plt.ylabel("Number of Customers")
# plt.show()

# correlation = df.corr(numeric_only=True)
# print(correlation)
# print()

# print("Plotting Correlation Heatmap")
# sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

# plt.figure(figsize=(7,4))
# sns.boxplot(x=df["Age"], color="lightgreen")
# plt.title("Boxplot of Customer Age")
# plt.xlabel("Age")
# plt.show()

print("Find the outliers in age")
outliers = df[df["Age"] > 100]
print("Found Outliers(s):")
print(outliers)