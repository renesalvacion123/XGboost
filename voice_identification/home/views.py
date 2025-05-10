import librosa
from django.core.files.storage import FileSystemStorage
import os
from django.shortcuts import render
import joblib
from . import utils

import os

# Get the absolute path to the "home" directory
HOME_DIR = os.path.dirname(os.path.abspath(__file__))

# Create a full absolute path to "home/uploaded_audio"
UPLOAD_DIR = 'home/uploaded_audio/'

# Ensure the directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)



# Load the model once when the view is called
def load_model():
    model_path = 'voice_classifier_model.pkl'  # Path to the saved model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def index(request):
    prediction = None
    model = load_model()  # Load the model here

    if request.method == 'POST' and 'audio_file' in request.FILES:
        audio_file = request.FILES['audio_file']
        
        # Save the uploaded file to the server
        fs = FileSystemStorage(location=UPLOAD_DIR)
        file_path = fs.save(audio_file.name, audio_file)
        full_file_path = os.path.join(UPLOAD_DIR, file_path)
        
        try:
            # Attempt to load the audio with librosa
            y, sr = librosa.load(full_file_path, sr=None)
            
            # Classify the uploaded file if the model is loaded
            if model:
                prediction = utils.classify_voice(full_file_path, model)  # Use the model to classify the audio
            else:
                prediction = "Error: Model failed to load."
        
        except Exception as e:
            print(f"Error loading audio file: {e}")
            prediction = "Error: Unable to load audio file"
        
        # Optionally, remove the file after classification
        os.remove(full_file_path)

    # Render the index page and pass the prediction to the context
    return render(request, 'home/index.html', {'prediction': prediction})
