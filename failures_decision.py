from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Reading the data
d = pd.read_csv("data/classification/failures.csv")

# Constructing the X and Y matrices
x = d[["falures_same_nuc", "pass_other_nuc", "failures_new_app", "pass_old_app"]]
y = d["decision"].values.tolist()

clf = DecisionTreeClassifier()
clf.fit(x, y)

data = {
  "falures_same_nuc": 1,
  "pass_other_nuc": 1,
  "failures_new_app": 0,
  "pass_old_app": 1
}

df = pd.DataFrame(data, index=[0])
prediction = clf.predict(df)
print(prediction[0])
