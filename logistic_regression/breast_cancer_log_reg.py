import pandas as pd 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("data/data.csv")
data.head()

data.info()
data.describe()


# Clean data
sns.heatmap(data.isnull())

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.head()

data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
data["diagnosis"] = data["diagnosis"].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")


# Select variables
y = data["diagnosis"]   # target var
x = data.drop(["diagnosis"], axis=1, inplace=False) # predictor var


# Normalize data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, random_state=42)


# Train model
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)


# Evaluation model
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy: .2f}")

print(classification_report(y_test, y_predict))

