import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample dataset (can expand)
data = {
    "Crop": ["Rice","Rice","Wheat","Wheat","Maize","Maize"],
    "N": [90,40,80,30,70,20],
    "P": [40,20,35,15,30,10],
    "K": [40,20,30,15,25,10],
    "Fertilizer": ["Urea","Compost","DAP","Compost","NPK","Compost"]
}

df = pd.DataFrame(data)

df["Crop"] = df["Crop"].astype("category").cat.codes

X = df[["Crop","N","P","K"]]
y = df["Fertilizer"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(df["Crop"].astype("category").cat.categories, "crop_labels.pkl")

print("✅ Fertilizer model trained")