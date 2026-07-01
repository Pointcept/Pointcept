import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("../data/dental_landmarks_mesh/labels.csv", dtype=str)

tasks = [
    "CLASSE DX",
    "CLASSE SX",
    "MORSO ANTERIORE",
    "TRASVERSALE (senza id denti)",
    "LINEE MEDIANE"
]
class_names = {
    "CLASSE DX": ["prima classe", "seconda classe", "terza classe"],
    "CLASSE SX": ["prima classe", "seconda classe", "terza classe"],
    "MORSO ANTERIORE": ["normale", "profondo", "aperto", "inverso"],
    "TRASVERSALE (senza id denti)": ["normale", "scissor", "cross"],
    "LINEE MEDIANE": ["centrata", "deviata"],
}

class_weights = []

print("\n=== Class Weights by Task ===\n")

for task in tasks:
    print(f"Task: {task}")
    labels = df[task]
    labels = labels[labels.isin(class_names[task])]
    y = labels.values

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(class_names[task]),
        y=y
    )
    class_weights.append(weights.tolist())

    for label, weight in zip(class_names[task], weights):
        print(f"  - {label:<15} => {weight:.4f}")
    print()
