import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("forestfires.csv")

# Create target column (fire risk: 1 if area > 0 else 0)
data['fire_risk'] = data['area'].apply(lambda x: 1 if x > 0 else 0)

# Rename RH → humidity for consistency
data = data.rename(columns={'RH': 'humidity'})

# Use only 4 features (names match your form)
X = data[['temp', 'humidity', 'wind', 'rain']]
y = data['fire_risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "forest_fire_model.pkl")
print("✅ Model saved as forest_fire_model.pkl")
