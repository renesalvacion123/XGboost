import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Path to the CSV file
csv_path = 'extracted_features_unbalanced.csv'

# Load the dataset
df = pd.read_csv(csv_path)

# Extract features (drop filename and label columns)
X = df.drop(columns=['filename', 'label']).values
y = df['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the model pipeline
model = make_pipeline(
    StandardScaler(),
    XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Testing Accuracy: {model.score(X_test, y_test):.4f}")

# Save the trained model
joblib.dump(model, 'voice_classifier_model.pkl')
print("✅ Model trained and saved as 'voice_classifier_model.pkl'")


print(df['label'].value_counts())
