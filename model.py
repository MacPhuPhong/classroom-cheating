import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset CSV file (sửa lại đường dẫn cho đúng)
df = pd.read_csv(r'D:\git&github\classroom-cheating\dataset_path\dataset_pathdataset.csv', low_memory=False)

# Xóa các dòng không có label
df = df.dropna(subset=['label']).reset_index(drop=True)

# Ép các cột số về dạng numeric, thay NaN bằng 0
for col in df.columns:
    if col not in ['image_name', 'label']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# Prepare feature matrix X and target vector y
X = df.drop(['label', 'image_name'], axis=1)  
y = df['label'].map({'Suspicious': 0, 'Normal': 1})  

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and configure the XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=50,              # Number of trees
    eval_metric='logloss',        # Evaluation metric
    objective='binary:logistic',  # Binary classification
    tree_method='hist',           # Fast histogram method
    eta=0.1,                      # Learning rate
    max_depth=3                   # Max tree depth
)

# Train the model
model.fit(X_train, y_train)

# Output the model details
print(model)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the trained model to a file
model.save_model(r"D:\git&github\classroom-cheating\trained_model.json")
