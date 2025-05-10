# train_model.py

import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utils import extract_features, is_audio_file  # Assuming utility functions are in a file named utils.py

# Set path to your dataset directory
dataset_path = 'archive'  # Replace this with your actual dataset path

# Lists to hold the feature vectors and labels
audio_files = []
labels = []

# Loop through the dataset and process each audio file
for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        file_path = os.path.join(root, filename)

        # Skip non-audio files
        if not is_audio_file(filename):
            continue

        # Check if the folder name contains 'fake' (AI-generated) or 'real' (Human-generated)
        if "fake" in root.lower():  # Fake folder (AI-generated)
            label = 1
        elif "real" in root.lower():  # Real folder (Human-generated)
            label = 0
        else:
            continue  # Skip files in other folders

        # Extract features from the audio file
        features = extract_features(file_path)
        audio_files.append(features)
        labels.append(label)

# Convert lists to numpy arrays
X = np.array(audio_files)
y = np.array(labels)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a StandardScaler and a RandomForestClassifier
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model (optional)
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Testing Accuracy: {model.score(X_test, y_test)}")

# Save the trained model to a file
joblib.dump(model, 'voice_classifier_model.pkl')

print("Model trained and saved as 'voice_classifier_model.pkl'")
