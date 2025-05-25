import os
import time
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import yt_dlp
import subprocess

matplotlib.use('Agg')  # Use safe backend for plotting


# ----------------------------- Feature Set -----------------------------
selected_features = [
"centroid_min",
"bandwidth_min",
"rolloff_min",
"centroid_mean",
"zcr_mean",
"mfcc2_mean",


]

def is_valid_audio_format(filename):
    valid_formats = ['.wav', '.mp3', '.flac']
    return any(filename.lower().endswith(ext) for ext in valid_formats)

def preprocess_audio(audio_path, target_sr=22050, max_duration=120):
    y, sr = librosa.load(audio_path, sr=None, mono=False, duration=max_duration)
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
    return y, target_sr


# ----------------------------- Feature Extraction -----------------------------
def extract_features(audio_path, sr=22050):
    y, sr = preprocess_audio(audio_path, target_sr=sr)
# Basic feature extraction
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_changes = np.count_nonzero(np.diff(np.argmax(pitches, axis=0)))

    # Construct feature vector dictionary
    feature_vector = {


        # ZCR features
        "zcr_mean": np.mean(zcr),


        # Spectral centroid
        "centroid_min": np.min(centroid),
        "centroid_mean": np.mean(centroid),

        # Rolloff
        "rolloff_min": np.min(rolloff),

        # Bandwidth
        "bandwidth_min": np.min(bandwidth),

        # MFCCs
        "mfcc2_mean": np.mean(mfcc[1]),



    }


    return [feature_vector[feat] for feat in selected_features]

def classify_voice(audio_path, model):
    try:
        features = extract_features(audio_path)
        print(f"Extracted features: {features}")

        if len(features) == 0:
            return "Error extracting features", 0

        prediction_proba = model.predict_proba([features])[0]
        label_index = prediction_proba.argmax()
        label = model.classes_[label_index]
        confidence = round(prediction_proba[label_index] * 100, 2)
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Error in classification", 0

    print(f"Model prediction: {label}, Confidence: {confidence}")
    return ("Low confidence", confidence) if confidence < 60 else (label, confidence)

def save_spectrogram(audio_path, spectrogram_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # Generate Mel spectrogram in decibels
    S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=1024, hop_length=1024, n_mels=64  # lighter
)

    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Time")
    plt.ylabel("Frequency (Mel)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
    plt.savefig(spectrogram_path)
    plt.close()

    print("Saved spectrogram to:", spectrogram_path)

def download_tiktok_video(url, output_path='tiktok_video.mp4'):
    # Universal options for yt-dlp that work across multiple platforms
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',  # Force best quality
        'quiet': False,
        'noplaylist': True,
        'merge_output_format': 'mp4',          # Ensure merged output
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print("Downloading video...")
            ydl.download([url])
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading video: {e}")

    return output_path


def trim_video_to_2_minutes(input_video_path, output_video_path='trimmed_video.mp4'):
    print("Trimming video to the first 2 minutes...")
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if exists
        '-i', input_video_path,  # Input file path
        '-t', '00:02:00',  # Trim to 2 minutes (00:02:00)
        '-c', 'copy',  # Copy audio and video streams without re-encoding
        output_video_path  # Output file path
    ]

    try:
        # Run the ffmpeg command and capture stdout and stderr
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if ffmpeg output anything to stdout or stderr
        print("FFmpeg process completed.")
        print(f"STDOUT: {result.stdout.decode()}")
        print(f"STDERR: {result.stderr.decode()}")

        if result.returncode == 0:
            print(f"Video trimmed to {output_video_path}")
            return output_video_path
        else:
            print(f"Error in trimming video: {result.stderr.decode()}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e.stderr.decode()}")
        return None




def extract_audio_with_ffmpeg(video_path, output_audio_path='output.wav'):
    print("Extracting audio with ffmpeg...")
    command = [
        'ffmpeg',
        '-y',  # Overwrite the output file if it exists
        '-i', video_path,  # Input video path
        '-vn',  # No video output
        '-acodec', 'pcm_s16le',  # Audio codec (WAV)
        '-ar', '22050',  # Set sample rate
        '-ac', '1',  # Mono audio (adjust if stereo is needed)
        output_audio_path  # Output path for the audio
    ]
    
    try:
        # Run the subprocess and capture stdout and stderr
        print(f"Running command: {' '.join(command)}")  # Debugging: print the command being run
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # If the subprocess is successful, print stdout and stderr
        print("Audio extraction completed.")
        print(f"STDOUT: {result.stdout.decode()}")  # Show the stdout (standard output)
        print(f"STDERR: {result.stderr.decode()}")  # Show the stderr (standard error)
        
        # Return the output path
        return output_audio_path

    except subprocess.CalledProcessError as e:
        # If there's an error, catch it and print both the error and the stderr
        print(f"Error extracting audio: {e.stderr.decode()}")
        return None
    except Exception as e:
        # Catch any other exception and print the error message
        print(f"Unexpected error: {str(e)}")
        return None


def tiktok_to_wav(url):
    try:
        # Step 1: Download the full video
        video_file = download_tiktok_video(url)
        if not os.path.exists(video_file):
            print("Error: Downloaded video file does not exist.")
            return None

        # Step 2: Trim the video to the first 2 minutes
        trimmed_video_file = trim_video_to_2_minutes(video_file)
        if not trimmed_video_file:
            print("Error: Failed to trim video.")
            return None

        # Step 3: Extract audio from the trimmed video
        wav_file = extract_audio_with_ffmpeg(trimmed_video_file)
        if not wav_file or not os.path.exists(wav_file):
            print("Error: Failed to extract audio.")
            return None

        try:
            # Step 4: Process the audio using librosa
            y, sr = librosa.load(wav_file, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Audio duration: {duration} seconds")

            if duration == 0:
                print("Error: Audio file is empty or silent.")
                return None
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

        # Step 5: Retry-safe deletion of the video file
        for i in range(5):
            try:
                os.remove(video_file)
                os.remove(trimmed_video_file)
                print(f"Deleted video files: {video_file} and {trimmed_video_file}")
                break
            except PermissionError:
                print("Video file still in use, retrying...")
                time.sleep(1)

        return wav_file

    except Exception as e:
        print(f"Error: {e}")
        return None


def clean_old_spectrograms(folder_path, max_age_seconds=30):
    now = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.png'):
            if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Deleted old spectrogram: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")