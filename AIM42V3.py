import os
import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pedalboard import Pedalboard, Compressor, Reverb, PeakFilter
from transformers import AutoProcessor, MusicgenForConditionalGeneration

st.set_page_config(layout="wide")

# Compact styling
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-baseweb="select"] {border-radius: 4px !important;}
        .stSlider {height: 40px;}
        div[role="radiogroup"] > label {flex: 1;}
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🎹 M42 - V0.3")
st.caption("Professional AI Music Generation with Kodály-inspired Features")

# Model selection in top bar
with st.container():
    col_model, col_space = st.columns([2, 8])
    with col_model:
        models = {
            "HuggingFace": "facebook/musicgen-small",
            "CarloLocal": "/home/gringo/Desktop/Msc/models/facebook/musicgen-small/"
        }
        model_selection = st.selectbox("Model", options=list(models.keys()))

# Load model and processor first
try:
    device = torch.device("cpu")
    model = MusicgenForConditionalGeneration.from_pretrained(models[model_selection]).to(device)
    processor = AutoProcessor.from_pretrained(models[model_selection])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def generate_music():
    # Access all UI components through st.session_state
    params = st.session_state
    full_prompt = f"{params.description} {', '.join(params.instruments)}. {params.scale_type} scale, " \
                f"{params.progression_type} progression, {params.mode_type} mood."

    if params.use_kodaly:
        full_prompt += f" {params.kodaly_rhythm} rhythm, {params.kodaly_form} structure, " \
                     f"emphasizing {params.kodaly_steps}% stepwise motion."

    inputs = processor(text=[full_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    
    try:
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=params.guidance_scale,
                max_new_tokens=int(params.desired_seconds * params.max_new_tokens_factor),
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p
            ).detach().cpu().numpy().ravel()
        
        # Normalize and save audio
        audio_values = audio_values / np.max(np.abs(audio_values)) if np.max(np.abs(audio_values)) != 0 else audio_values
        output_file = "musicgen_output.wav"
        wavfile.write(output_file, 32000, (audio_values * 32767.0).astype(np.int16))

        # Apply audio processing
        board = Pedalboard([])
        if params.apply_reverb:
            board.append(Reverb(room_size=params.reverb_room_size))
        if params.apply_compression:
            board.append(Compressor(ratio=params.compression_ratio))
        if params.apply_eq:
            board.append(PeakFilter(gain_db=params.eq_low_gain, cutoff_frequency_hz=100))
            board.append(PeakFilter(gain_db=params.eq_mid_gain, cutoff_frequency_hz=1000))
            board.append(PeakFilter(gain_db=params.eq_high_gain, cutoff_frequency_hz=5000))

        processed_audio = board(audio_values, 32000)
        wavfile.write(output_file, 32000, (processed_audio * 32767.0).astype(np.int16))

        # Update session state
        st.session_state.audio_generated = True
        st.session_state.audio_file = output_file
        
    except Exception as e:
        st.error(f"Music generation error: {e}")
        st.session_state.audio_generated = False

# Main grid layout
main_col1, main_col2 = st.columns([2, 1], gap="large")

with main_col1:
    # Left Panel: Core Parameters
    with st.container(border=True):
        st.subheader("🎼 Composition Setup")
        
        # Music parameters in columns
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.music_type = st.selectbox("Genre", ["Classical", "Jazz", "Rock", "Electronic", "Cinematic", 
                                                              "Ambient", "Folk", "Experimental", "Hiphop", "Rap", 
                                                              "Soul", "Kodály Folk"])
            st.session_state.scale_type = st.selectbox("Scale", ["Major", "Minor", "Pentatonic (Kodály)", 
                                                               "Harmonic Minor", "Dorian", "Phrygian", "Lydian", 
                                                               "Mixolydian", "Blues"])
            
        with col2:
            st.session_state.progression_type = st.selectbox("Progression", ["I-IV-V", "ii-V-I", "I-V-vi-IV", 
                                                                           "I-vi-ii-V", "Custom...", "Modal", 
                                                                           "Atonal", "Chromatic", "Kodály Folk (Stepwise)"])
            st.session_state.mode_type = st.selectbox("Mood", ["Happy", "Melancholic", "Energetic", "Calm", 
                                                             "Mysterious", "Dramatic", "Playful", "Somber"])

        # Instruments and duration
        st.session_state.instruments = st.multiselect("Instruments", ["Piano", "Strings", "Electric Guitar", "Drums", 
                                                                     "Synthesizer", "Brass", "Woodwinds", "Harp", 
                                                                     "Bass808"], default=["Piano"], max_selections=4)
        st.session_state.desired_seconds = st.slider("Duration (seconds)", 10, 300, 30)

    # Advanced Parameters Grid
    with st.container(border=True):
        st.subheader("⚙️ Advanced Controls")
        
        tab_gen, tab_proc = st.tabs(["Generation", "Processing"])
        
        with tab_gen:
            col_gen1, col_gen2 = st.columns(2)
            with col_gen1:
                st.session_state.temperature = st.slider("Creativity", 0.1, 2.0, 0.8, help="Higher values = more random")
                st.session_state.top_k = st.slider("Top-K", 1, 100, 50)
                
            with col_gen2:
                st.session_state.top_p = st.slider("Nucleus Sampling", 0.1, 1.0, 0.95)
                st.session_state.guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 8.0)
            
            st.session_state.max_new_tokens_factor = st.slider("Duration Factor", 10, 30, 20)

        with tab_proc:
            col_proc1, col_proc2 = st.columns(2)
            with col_proc1:
                st.session_state.apply_reverb = st.checkbox("Reverb", True)
                st.session_state.reverb_room_size = st.slider("Room Size", 0.0, 1.0, 0.5)
                
            with col_proc2:
                st.session_state.apply_compression = st.checkbox("Compression", True)
                st.session_state.compression_ratio = st.slider("Ratio", 1.0, 10.0, 4.0)
                
            st.session_state.apply_eq = st.checkbox("Equalizer", True)
            eq_low, eq_mid, eq_high = st.columns(3)
            with eq_low: st.session_state.eq_low_gain = st.slider("Bass", -12, 12, 0)
            with eq_mid: st.session_state.eq_mid_gain = st.slider("Mid", -12, 12, 0)
            with eq_high: st.session_state.eq_high_gain = st.slider("Treble", -12, 12, 0)

with main_col2:
    # Right Panel: Generation Interface
    with st.container(border=True):
        st.subheader("🚀 Generation Interface")
        
        st.session_state.description = st.text_area("Prompt", 
            f"A {st.session_state.music_type} composition in {st.session_state.scale_type} scale with " \
            f"{st.session_state.progression_type} progression, evoking a {st.session_state.mode_type} mood.", 
            height=100)
        
        # Kodály Features
        st.session_state.use_kodaly = st.checkbox("Enable Kodály Features", False)
        if st.session_state.use_kodaly:
            col_kod1, col_kod2 = st.columns(2)
            with col_kod1:
                st.session_state.kodaly_rhythm = st.selectbox("Rhythm", ["ta (Quarter)", "ti-ti (Eighth)", 
                                                                        "tika-tika (16th)", "Syncopation"])
                st.session_state.kodaly_form = st.selectbox("Structure", ["ABA", "AAB", "Call-Response"])
            with col_kod2:
                st.session_state.kodaly_steps = st.slider("Stepwise Motion", 50, 100, 80)
        
        # Generate button
        if st.button("🎵 Generate Music", use_container_width=True, type="primary"):
            generate_music()

    # Audio Output
    if st.session_state.get('audio_generated', False):
        with st.container(border=True):
            st.subheader("🎧 Output")
            st.audio(st.session_state.audio_file)
            
            with open(st.session_state.audio_file, "rb") as f:
                st.download_button(
                    "Download WAV",
                    data=f.read(),
                    file_name="composition.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
