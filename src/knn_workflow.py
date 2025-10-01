import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap
import os

# =============== Setup ===============
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Iris.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== 1. Load Data ===============
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, 1:5].values   # features: Sepal & Petal
y = df.iloc[:, 5].values     # target: Species

# =============== 2. Normalize Features ===============
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# =============== 3. Train-Test Split ===============
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============== 4. KNN with Different K ===============
accuracies = {}
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc

best_k = max(accuracies, key=accuracies.get)
print("Best k:", best_k, "with accuracy:", accuracies[best_k])

# =============== 5. Train Final Model ===============
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# Accuracy & Confusion Matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Final Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f"KNN (k={best_k}) Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# =============== 6. Decision Boundary Visualization ===============
# Use only 2 features (PetalLength, PetalWidth) for 2D boundary
X_vis = df.iloc[:, [3, 4]].values
y_vis = df.iloc[:, 5].values

# Encode target labels into numbers for plotting
le = LabelEncoder()
y_vis_encoded = le.fit_transform(y_vis)

X_vis_scaled = sc.fit_transform(X_vis)

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis_scaled, y_vis_encoded, test_size=0.2, random_state=42, stratify=y_vis_encoded
)

knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train_vis, y_train_vis)

x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
colors = ["#FFAAAA", "#AAFFAA", "#AAAAFF"]
cmap_background = ListedColormap(colors)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

# Plot training points (still with labels, not encoded)
for i, label in enumerate(le.classes_):
    plt.scatter(
        X_vis_scaled[y_vis_encoded == i, 0],
        X_vis_scaled[y_vis_encoded == i, 1],
        label=label, edgecolor="k", s=50
    )
plt.legend()
plt.title(f"KNN Decision Boundary (Petal Length vs Width, k={best_k})")
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.savefig(os.path.join(OUTPUT_DIR, "decision_boundary.png"))
plt.close()
