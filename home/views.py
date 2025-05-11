import os
import librosa
import joblib
import soundfile as sf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from . import utils

# Constants
UPLOAD_DIR = 'home/uploaded_audio/'
MAX_AUDIO_DURATION = 120  # 2 minutes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'FINAL_XGB_MODEL.joblib')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None
    

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model once
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# View for file upload and classification
def index(request):
    model = load_model()
    label = None
    confidence = None
    spectrogram_image = None

    if request.method == 'POST' and 'audio_file' in request.FILES:
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        file_path = fs.save(audio_file.name, audio_file)
        full_file_path = os.path.join(UPLOAD_DIR, file_path)

        try:
            # Check audio file format
            if not utils.is_valid_audio_format(audio_file.name):
                label = "Invalid audio format"
                confidence = 0
                return render(request, 'home/index.html', {
                    'label': label,
                    'confidence': confidence,
                    'prediction': False
                })

            # Load and trim if needed
            y, sr = librosa.load(full_file_path, sr=None)
            print(f"Audio loaded: {y.shape}, Sample rate: {sr}")
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Audio duration: {duration} seconds")

            if duration == 0:
                label = "Empty or silent audio"
                confidence = 0
                return render(request, 'home/index.html', {
                    'label': label,
                    'confidence': confidence,
                    'prediction': False
                })

            if duration > MAX_AUDIO_DURATION:
                y = y[:sr * MAX_AUDIO_DURATION]

            # Save trimmed audio
            trimmed_audio_path = os.path.join(UPLOAD_DIR, f'trimmed_{file_path}')
            sf.write(trimmed_audio_path, y, sr)

            # Clean old spectrograms
            utils.clean_old_spectrograms('static/spectrograms')

            # Generate and save spectrogram
            spectrogram_path = f'static/spectrograms/{file_path}.png'
            utils.save_spectrogram(trimmed_audio_path, spectrogram_path)
            spectrogram_image = spectrogram_path


            # Classify using model
            if model:
                label, confidence = utils.classify_voice(trimmed_audio_path, model)
                print(f"Model loaded: {model is not None}")
                print(f"Label: {label}, Confidence: {confidence}")

                if label == 1:
                    label = "FAKE"
                elif label == 0:
                    label = "REAL"
                else:
                    label = "Unknown"
            else:
                label = "Error"
                confidence = 0

        except Exception as e:
            print(f"Error loading audio: {e}")
            label = "Error loading audio"
            confidence = 0

        # Cleanup
        os.remove(full_file_path)
        os.remove(trimmed_audio_path)

    return render(request, 'home/index.html', {
        'label': label,
        'confidence': confidence,
        'spectrogram_image': spectrogram_image,
        'prediction': label is not None and label not in ["Error", "Error loading audio"]
    })

def tiktok_audio_analysis(request):
    model = load_model()
    print(f"Model loaded: {model}")  # Debug print
    label = None
    confidence = None
    spectrogram_image = None

    if request.method == 'POST':
        tiktok_url = request.POST.get('tiktok_url')
        if tiktok_url:
            wav_file = utils.tiktok_to_wav(tiktok_url)
            
            # Check if the WAV file exists
            if wav_file and os.path.exists(wav_file):
                try:
                    print(f"Audio file {wav_file} exists and is valid.")

                    # Load and trim the audio file
                    y, sr = librosa.load(wav_file, sr=None)
                    print(f"Audio loaded: {y.shape}, Sample rate: {sr}")
                    duration = librosa.get_duration(y=y, sr=sr)
                    print(f"Audio duration: {duration} seconds")

                    if duration == 0:
                        label = "Empty or silent audio"
                        confidence = 0
                        return render(request, 'home/index.html', {
                            'label': label,
                            'confidence': confidence,
                            'prediction': False
                        })

                    # Trim the audio if it exceeds the max duration
                    if duration > MAX_AUDIO_DURATION:
                        y = y[:sr * MAX_AUDIO_DURATION]

                    # Save the trimmed audio
                    trimmed_audio_path = f'{UPLOAD_DIR}trimmed_{os.path.basename(wav_file)}'
                    sf.write(trimmed_audio_path, y, sr)

                    # Clean old spectrograms
                    utils.clean_old_spectrograms('static/spectrograms')

                    # Generate and save spectrogram
                    filename = os.path.basename(wav_file).replace('.wav', '')
                    spectrogram_path = f'static/spectrograms/tiktok_{filename}.png'

                    utils.save_spectrogram(trimmed_audio_path, spectrogram_path)
                    spectrogram_image = spectrogram_path


                    # Classify using the model
                    if model:
                        print(f"Classifying audio with model: {model}")  # Debug print
                        label, confidence = utils.classify_voice(trimmed_audio_path, model)
                        print(f"Prediction: {label}, Confidence: {confidence}")  # Debug print
                        if label == 1:
                            label = "FAKE"
                        elif label == 0:
                            label = "REAL"
                    else:
                        label = "Error"
                        confidence = 0

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    label = "Error processing audio"
                    confidence = 0
                finally:
                    os.remove(trimmed_audio_path)

    return render(request, 'home/index.html', {
        'label': label,
        'confidence': confidence,
        'spectrogram_image': spectrogram_image,
        'prediction': label is not None and label not in ["Error", "Error processing audio"]
    })
