import os
import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pedalboard import Pedalboard, Compressor, Limiter, Reverb, HighpassFilter, PeakFilter, Gain, LowpassFilter, Distortion, Chorus, Delay
from transformers import AutoProcessor, MusicgenForConditionalGeneration

st.set_page_config(layout="wide")

# Streamlit UI Setup
st.title("M42 - V0.1")
st.write("Professional-grade AI Music Generation with Detailed Controls")

# Available models
models = {
    "HuggingFace": "facebook/musicgen-small",
    "CarloLocal": "/home/gringo/Desktop/Msc/models/facebook/musicgen-small/"
}

# Load the model and processor
model_selection = st.selectbox("Select Model", options=list(models.keys()))
selected_model_path = models[model_selection]

# Function to check if necessary files exist in the local directory
def check_local_model_files(model_path):
    required_files = [
        "config.json", "pytorch_model.bin", "preprocessor_config.json"
    ]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    return missing_files

# Check for missing files in the local directory
missing_files = []
if selected_model_path != "facebook/musicgen-small":
    missing_files = check_local_model_files(selected_model_path)

# Fallback to Hugging Face hosted version if files are missing
if missing_files:
    st.warning(f"Missing the following files in the local directory: {', '.join(missing_files)}. Falling back to Hugging Face hosted version.")
    selected_model_path = "facebook/musicgen-small"

# Load model and processor
try:
    model = MusicgenForConditionalGeneration.from_pretrained(selected_model_path).to(torch.device("cpu"))
    processor = AutoProcessor.from_pretrained(selected_model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# ====== Musical Parameters Section ======
st.header("Musical Composition Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    music_type = st.selectbox("Music Type", ["Classical", "Jazz", "Rock", "Electronic", "Cinematic", "Ambient", "Folk", "Experimental", "Hiphop", "Rap", "Soul"])

with col2:
    scale_type = st.selectbox("Scale", ["Major", "Minor", "Harmonic Minor", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Blues"])

with col3:
    progression_type = st.selectbox("Progression", [
        "I-IV-V", "ii-V-I", "I-V-vi-IV", "I-vi-ii-V", 
        "Custom...", "Modal", "Atonal", "Chromatique"
    ])

with col4:
    mode_type = st.selectbox("Mode", [
        "Happy", "Melancholic", "Energetic", "Calm", 
        "Mysterious", "Dramatic", "Playful", "Somber"
    ])

# ====== Advanced Generation Controls ======
with st.expander("Advanced Generation Parameters"):
    col_gen1, col_gen2, col_gen3 = st.columns(3)
    
    with col_gen1:
        temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.8)
        top_k = st.slider("Top-K Sampling", 1, 100, 50)
        
    with col_gen2:
        top_p = st.slider("Nucleus Sampling (Top-P)", 0.1, 1.0, 0.95)
        guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 8.0)
        
    with col_gen3:
        max_new_tokens_factor = st.slider("Duration Factor", 10, 30, 20,
            help="Tokens per second (affects generation length)")

# ====== Audio Processing Controls ======
with st.expander("Advanced Audio Processing Parameters"):
    proc_col1, proc_col2, proc_col3 = st.columns(3)
    
    with proc_col1:
        processing_params = {
            'low_cut': st.slider("Low Cut (Hz)", 20, 500, 100),
            'high_cut': st.slider("High Cut (Hz)", 5000, 20000, 15000),
            'compression_threshold': st.slider("Compression Threshold (dB)", -30, 0, -10),
            'compression_ratio': st.slider("Compression Ratio", 1.0, 10.0, 3.0)
        }
        
    with proc_col2:
        processing_params.update({
            'attack_time': st.slider("Attack Time (ms)", 1, 100, 10),
            'release_time': st.slider("Release Time (ms)", 50, 1000, 100),
            'high_boost_gain': st.slider("High Boost Gain (dB)", 0, 12, 3),
            'high_boost_freq': st.slider("High Boost Freq (Hz)", 4000, 12000, 8000)
        })
        
    with proc_col3:
        processing_params.update({
            'reverb_wet': st.slider("Reverb Mix", 0.0, 1.0, 0.3),
            'delay_time': st.slider("Delay Time (s)", 0.1, 1.0, 0.3),
            'distortion_drive': st.slider("Distortion Drive", 0, 24, 10),
            'final_limiter_threshold': st.slider("Limiter Threshold (dB)", -12, 0, -3)
        })

# ====== Main Generation Interface ======
st.header("Generation Parameters")
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    description = st.text_area("Musical Description", 
        f"A {music_type} composition in {scale_type} scale with {progression_type} progression, "
        f"evoking a {mode_type} mood. Featuring: ")
    
with main_col2:
    desired_seconds = st.slider("Duration (seconds)", 10, 300, 30)
    instruments = st.multiselect("Instruments", [
        "Piano", "Strings", "Electric Guitar", "Drums", 
        "Synthesizer", "Brass", "Woodwinds", "Harp", "Bass808", 
    ], default=["Piano"])
    
    enable_processing = st.checkbox("Enable Audio Post-Processing", True)

def apply_audio_processing(input_file, output_file, params):
    try:
        # Load the audio file
        sample_rate, audio_data = wavfile.read(input_file)

        # Convert audio to float32 for processing (normalize to [-1.0, 1.0])
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Ensure proper audio shape (mono to stereo if needed)
        if len(audio_data.shape) == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        # Create a Pedalboard with effects
        board = Pedalboard([
            HighpassFilter(params['low_cut']),
            LowpassFilter(params['high_cut']),
            Compressor(
                threshold_db=params['compression_threshold'],
                ratio=params['compression_ratio'],
                attack_ms=params['attack_time'],
                release_ms=params['release_time']
            ),
            PeakFilter(
                cutoff_frequency_hz=params['high_boost_freq'],
                gain_db=params['high_boost_gain']
            ),
            Reverb(
                room_size=0.5,  # Fixed room size
                wet_level=params['reverb_wet']  # Correct parameter for mix
            ),
            Delay(delay_seconds=params['delay_time']),
            Distortion(drive_db=params['distortion_drive']),
            Limiter(threshold_db=params['final_limiter_threshold'])
        ])

        # Apply the effects
        processed_audio = board.process(audio_data, sample_rate)

        # Normalize and convert back to int16
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        processed_audio = (processed_audio * 32767.0).astype(np.int16)

        # Save the processed file
        wavfile.write(output_file, sample_rate, processed_audio)
        return True

    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return False

def generate_music():
    # Construct the full prompt from the description and selected instruments
    full_prompt = f"{description} {', '.join(instruments)}. Technical parameters: {scale_type} scale, {progression_type} progression, {mode_type} mood."
    
    # Process the text input using the AutoProcessor for tokenization
    inputs = processor(
        text=[full_prompt],  # Pass the text prompt as a list
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Calculate the number of tokens based on the desired duration
    max_new_tokens = int(desired_seconds * max_new_tokens_factor)
    
    try:
        # Generate music using the model
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,  # Pass the tokenized inputs to the model
                do_sample=True,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
        # Process the audio output
        audio = audio_values[0].cpu().numpy().ravel()
        
        # Normalize the audio to avoid clipping
        if np.max(np.abs(audio)) == 0:
            st.warning("Generated audio is silent - trying again...")
            audio = np.random.uniform(-0.1, 0.1, len(audio))

        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32767.0).astype(np.int16)
        
        # Save the generated audio to a file
        output_file = "musicgen_output.wav"
        wavfile.write(output_file, 32000, audio)
        
        # Apply post-processing if enabled
        if enable_processing:
            processed_file = "processed_output.wav"
            if apply_audio_processing(output_file, processed_file, processing_params):
                st.audio(processed_file, format='audio/wav')
                st.download_button("Download Processed Audio", processed_file)
            else:
                st.audio(output_file, format='audio/wav')
        else:
            st.audio(output_file, format='audio/wav')
            st.download_button("Download Raw Audio", output_file)
            
        st.success("Generation complete!")
        
    except Exception as e:
        st.error(f"Music generation error: {e}")

# Generate music on button click
if st.button("Generate Music"):
    generate_music()
