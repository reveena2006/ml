import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
data = load_breast_cancer()
X = data.data[:, :5]  
y = data.target
feature_names = data.feature_names[:5]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = SVC(kernel='rbf', C=2, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Model trained successfully!")
print("Accuracy:", accuracy)

class CancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Prediction (SVM - 5 Features)")
        self.entries = []

        tk.Label(root, text="Enter Patient Details", font=("Arial", 14)).pack(pady=10)

        # Input Fields
        for i, feature in enumerate(feature_names):
            frame = tk.Frame(root)
            frame.pack(pady=5)

            tk.Label(frame, text=feature, width=20, anchor='w').pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=20)
            entry.pack(side=tk.RIGHT)
            self.entries.append(entry)

        
        tk.Button(root, text="Predict", command=self.predict, bg="blue", fg="white").pack(pady=5)
        tk.Button(root, text="Show Accuracy", command=self.show_accuracy).pack(pady=5)
        tk.Button(root, text="Show Confusion Matrix", command=self.show_confusion).pack(pady=5)
        tk.Button(root, text="Show Graph", command=self.show_graph).pack(pady=5)
        tk.Button(root, text="Auto Fill Sample", command=self.autofill).pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def predict(self):
        try:
            values = [float(entry.get()) for entry in self.entries]
            input_data = np.array(values).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            if prediction == 1:
                result = "Benign (Non-Cancerous)"
                color = "green"
                confidence = probability[1]
            else:
                result = "Malignant (Cancerous)"
                color = "red"
                confidence = probability[0]

            self.result_label.config(
                text=f"Prediction: {result}\nConfidence: {confidence:.2f}",
                fg=color
            )

        except ValueError:
            messagebox.showerror("Error", "Enter valid numeric values!")

    def show_accuracy(self):
        messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.4f}")

    def show_confusion(self):
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def show_graph(self):
       
        X_vis = data.data[:, :2]
        y_vis = data.target

        scaler_vis = StandardScaler()
        X_vis_scaled = scaler_vis.fit_transform(X_vis)

        model_vis = SVC(kernel='rbf', C=2, gamma='scale')
        model_vis.fit(X_vis_scaled, y_vis)

        x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
        y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model_vis.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(7,6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis, cmap='coolwarm', edgecolors='k')

        plt.xlabel("Mean Radius (scaled)")
        plt.ylabel("Mean Texture (scaled)")
        plt.title("SVM Decision Boundary (2 Features)")
        plt.colorbar(label="Class (0=Malignant, 1=Benign)")
        plt.show()

    def autofill(self):
        sample = X[0]
        for i in range(5):
            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, str(sample[i]))




root = tk.Tk()
app = CancerApp(root)
root.mainloop()