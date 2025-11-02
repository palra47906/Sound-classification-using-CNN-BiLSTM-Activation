import os
import urllib.request
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import soundfile as sf
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from audiomentations import Compose
import io  # Required for in-memory file handling
from pydub import AudioSegment  # Required for MP3 conversion

# --- Page Config ---
st.set_page_config(page_title="Urban Sound Classifier 🎷", layout="wide")

# --- FIX #1: Added the WarmUpCosine class definition ---
# Keras needs this to load your trained model
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, warmup_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr

    def __call__(self, step):
        pi = tf.constant(np.pi, dtype=tf.float32)
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        base_lr = tf.cast(self.base_lr, tf.float32)
        warmup_lr = tf.cast(self.warmup_lr, tf.float32)

        cosine_decay = 0.5 * base_lr * (1 + tf.cos(pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        linear_warmup = warmup_lr + (base_lr - warmup_lr) * (step / warmup_steps)

        return tf.cond(step < warmup_steps, lambda: linear_warmup, lambda: cosine_decay)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_lr": self.warmup_lr
        }

# --- NEW FEATURE: Class Descriptions Dictionary ---
CLASS_DESCRIPTIONS = {
    "air_conditioner": "The low, humming sound of an air conditioning unit.",
    "car_horn": "A brief, loud, and typically high-pitched sound from a vehicle horn.",
    "children_playing": "Indistinct sounds of children's voices, laughter, and shouting during play.",
    "dog_bark": "The sharp, repeated vocalization of a dog.",
    "drilling": "A high-pitched, continuous, or intermittent buzzing sound from a power drill.",
    "engine_idling": "The low-frequency, rumbling sound of a vehicle's engine running while stationary.",
    "gun_shot": "A very short, loud, and sharp percussive sound.",
    "jackhammer": "A loud, repetitive, and percussive metallic impact sound.",
    "siren": "A loud, wailing, and high-pitched sound with a varying pitch, used by emergency vehicles.",
    "street_music": "Music (e.g., from a car stereo or busker) heard in an outdoor, urban environment."
}

# --- Get User Location, Date and Time ---
def get_user_location_and_time():
    try:
        response = requests.get("http://ip-api.com/json/")
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "Unknown City")
            region = data.get("regionName", "Unknown Region")
            country = data.get("country", "Unknown Country")
            timezone = data.get("timezone", "UTC")

            local_zone = pytz.timezone(timezone)
            now = datetime.now(local_zone)

            formatted_time = now.strftime("%I:%M %p")
            formatted_date = now.strftime("%A, %d %B %Y")

            return {
                "location": f"{city}, {region}, {country}",
                "date": formatted_date,
                "time": formatted_time
            }
    except:
        return {
            "location": "Unknown Location",
            "date": datetime.utcnow().strftime("%A, %d %B %Y"),
            "time": datetime.utcnow().strftime("%I:%M %p")
        }

user_info = get_user_location_and_time()

# --- Custom CSS ---
st.markdown(f"""
<style>
    .stButton>button {{
        color: white;
        background-color: #4CAF50;
        font-size: 18px;
        border-radius: 8px;
    }}
    .css-1aumxhk {{
        background-color: #f0f2f6;
    }}
    .weather-box {{
        position: absolute;
        top: 10px;
        right: 20px;
        text-align: right;
        font-size: 14px;
        color: #444;
    }}
    @media (max-width: 768px) {{
        .weather-box {{
            position: relative;
            text-align: left;
            font-size: 12px;
            padding: 10px;
        }}
        h1 {{
            font-size: 24px;
        }}
        .stButton>button {{
            font-size: 16px;
        }}
        .stMarkdown {{
            font-size: 14px;
        }}
    }}
</style>
<div class="weather-box">
    📍 {user_info['location']}<br>
    🕕 {user_info['date']}<br>
    🕒 {user_info['time']}
</div>
""", unsafe_allow_html=True)

# --- Load Class Mapping ---
@st.cache_data
def load_class_mapping():
    metadata_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/main/UrbanSound8K.csv"
    df = pd.read_csv(metadata_url)
    return dict(zip(df['classID'], df['class']))

class_mapping = load_class_mapping()

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    model_url = "https://huggingface.co/palra47906/Sound_Classification_Model_using_CNN_LSTM/resolve/main/Urbansound8K_CNN%2BLSTM_29102025.keras"
    
    # --- FIX #2: Changed to a relative path ---
    # This will save the model in the same folder as your app.py
    model_path = "Urbansound8K_CNN+LSTM_29102025.keras" 
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (This may take a minute)"):
            urllib.request.urlretrieve(model_url, model_path)
            
    # --- FIX #1 (continued): Pass custom_objects to load_model ---
    return load_model(model_path, custom_objects={"WarmUpCosine": WarmUpCosine})

# --- Feature Extraction ---
def extract_features(file, fixed_length=168):
    try:
        y, sr = librosa.load(file, sr=22050, mono=True)
        n_fft = min(2048, len(y))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=168, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_db.shape[1] > fixed_length:
            mel_db = mel_db[:, :fixed_length]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, fixed_length - mel_db.shape[1])), mode='constant')

        return mel_db, sr, y
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Prediction ---
@tf.function(reduce_retracing=True)
def make_prediction(model, input_tensor):
    return model(input_tensor, training=False)

def predict_class(model, file):
    features, sr, audio = extract_features(file)
    if features is None:
        return None, None, None, None

    features = (features - np.mean(features)) / np.std(features)
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

    prediction = make_prediction(model, input_tensor)
    predicted_class = np.argmax(prediction.numpy())
    confidence = np.max(prediction.numpy())
    label = class_mapping.get(predicted_class, "Unknown")

    return label, sr, audio, features, prediction.numpy()

# --- Streamlit App ---
st.markdown("""
    <h1 style='display: flex; align-items: center; gap: 10px;'>
        🎷 Urban Sound Classifier
        <img src='https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg' width='60'>
    </h1>
""", unsafe_allow_html=True)

st.markdown("Upload a `.wav` or `.mp3` file to predict the sound class using a CNN-LSTM-Transformer model trained on UrbanSound8K.")

model = load_trained_model()

# --- MODIFICATION: Accept .mp3 files ---
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Display the original audio file
    st.audio(uploaded_file, format=uploaded_file.type)

    # --- MODIFICATION: Add conversion logic ---
    file_to_process = None
    if uploaded_file.type == "audio/mpeg" or uploaded_file.name.endswith(".mp3"):
        with st.spinner("Converting MP3 to WAV..."):
            try:
                # Read MP3 file from upload
                audio_segment = AudioSegment.from_mp3(uploaded_file)
                
                # Create an in-memory WAV file
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0) # Rewind to the beginning of the file
                
                file_to_process = wav_io
                st.info("MP3 successfully converted to WAV for analysis.")
            except Exception as e:
                st.error(f"Error converting MP3: {e}")
                st.error("Please ensure FFmpeg is installed if running locally.")
                st.stop()
                
    elif uploaded_file.type == "audio/wav" or uploaded_file.name.endswith(".wav"):
        file_to_process = uploaded_file
    
    else:
        st.error("Unsupported file type. Please upload a WAV or MP3 file.")
        st.stop()
    # --- END MODIFICATION ---

    # --- SYNTAX FIX: This block was indented incorrectly ---
    if file_to_process:
        # Pass the (potentially converted) file to the prediction function
        label, sr, audio, features, prediction = predict_class(model, file_to_process)

        if label:
            st.success(f"✅ Predicted Class: **{label}**")
            
            # --- NEW FEATURE: Class Description ---
            with st.expander("What is this sound?"):
                description = CLASS_DESCRIPTIONS.get(label, "No description available for this class.")
                st.write(description)
            # --- END NEW FEATURE ---
            
            confidence = np.max(prediction)
            st.metric(label="Prediction Confidence", value=f"{confidence * 100:.2f}%")

            # --- Create two columns for charts ---
            col1, col2 = st.columns(2)

            with col1:
                # --- Top 3 Predictions ---
                st.subheader("Top 3 Predictions")
                top3_idx = np.argsort(prediction[0])[-3:][::-1]
                top3_labels = [class_mapping.get(i, "Unknown") for i in top3_idx]
                top3_scores = prediction[0][top3_idx]

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(top3_labels, top3_scores, color='skyblue')
                ax.invert_yaxis()
                ax.set_xlabel('Confidence', fontsize=10)
                ax.set_title('Top 3 Predictions', fontsize=12)
                ax.tick_params(axis='both', labelsize=9)
                st.pyplot(fig)
            
            with col2:
                # --- NEW FEATURE: Full Probability Breakdown ---
                st.subheader("All Class Probabilities")
                prob_df = pd.DataFrame({
                    "Class": [class_mapping.get(i, "Unknown") for i in range(len(prediction[0]))],
                    # --- FIX FOR TypeError: Convert numpy.float32 to standard float ---
                    "Probability": prediction[0].astype(float) 
                })
                prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df, 
                             # --- FIX: Replaced 'use_container_width' with 'width' ---
                             width='stretch',
                             column_config={
                                 "Probability": st.column_config.ProgressColumn(
                                     "Probability",
                                     format="%.2f%%",
                                     # --- FIX: Renamed arguments ---
                                     min_value=0,
                                     max_value=1
                                 )
                             })
                # --- END NEW FEATURE ---

            # --- Waveform Plot ---
            st.subheader("Waveform")
            fig_wave, ax_wave = plt.subplots(figsize=(8, 2))
            librosa.display.waveshow(audio, sr=sr, ax=ax_wave)
            ax_wave.set_title("Waveform", fontsize=12)
            ax_wave.set_xlabel("Time (s)", fontsize=10)
            ax_wave.set_ylabel("Amplitude", fontsize=10)
            ax_wave.tick_params(axis='both', labelsize=9)
            st.pyplot(fig_wave)

            # --- Mel Spectrogram ---
            st.subheader("Mel Spectrogram")
            colormap = st.selectbox("Select Color Map", ['viridis', 'plasma', 'inferno', 'magma'], index=0)

            fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
            
            # --- FIX: Correct way to create a colorbar ---
            # 1. specshow returns the "mappable" artist
            img = librosa.display.specshow(features[0, :, :, 0], sr=sr, x_axis='time', y_axis='mel', cmap=colormap, ax=ax_spec)
            # 2. Pass the mappable 'img' to the figure's colorbar method
            fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
            # --- END FIX ---

            ax_spec.set_title(f"Mel Spectrogram - {label}", fontsize=12)
            ax_spec.set_xlabel("Time (s)", fontsize=10)
            ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
            ax_spec.tick_params(axis='both', labelsize=9)
            st.pyplot(fig_spec)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Darlington+Signature&display=swap');
    </style>
    <center>Made with ❤️ by <span style="font-family: 'Darlington Signature', cursive;">Arijit Pal</span></center>
""", unsafe_allow_html=True)

