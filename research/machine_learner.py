import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ---------------------------
# Step 1: Generate random regression data
# ---------------------------
np.random.seed(42)
X = np.random.rand(2000, 5)  # 20 samples, 5 features
y = np.random.rand(2000)     # 20 target values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# ---------------------------
# Step 2: Train XGBoost Regressor
# ---------------------------
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to JSON
model.save_model("model.json")
print("Model saved as model.json")

# ---------------------------
# Step 3: Load model
# ---------------------------
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("model.json")

# ---------------------------
# Step 4: Make predictions
# ---------------------------
y_pred = loaded_model.predict(X_test)
print("\nPredictions on test set (first 5 samples):")
print(y_pred[:5])

# ---------------------------
# Step 5: Print first 5 input vectors in C array format
# ---------------------------
print("\nfloat test_cases[][5] = {")
for i in range(5):
    features = ", ".join(f"{x:.8f}" for x in X_test[i])
    print(f"    {{{features}}}, // Test {i+1}")
print("};")
