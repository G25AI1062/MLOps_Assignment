from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
faces = fetch_olivetti_faces()

X = faces.data
y = faces.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "savedmodel.pth")

print("Model trained successfully!")
print("Model saved as savedmodel.pth")