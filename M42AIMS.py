import os
import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pedalboard import Pedalboard, Compressor, Reverb, PeakFilter, Delay
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pretty_midi
from pydub import AudioSegment
from io import BytesIO



        

# Initialize session state
if 'tracks' not in st.session_state:
    st.session_state.tracks = []
if 'drum_pattern' not in st.session_state:
    st.session_state.drum_pattern = [[False]*16 for _ in range(4)]
if 'sequencer_notes' not in st.session_state:
    st.session_state.sequencer_notes = [[False]*16 for _ in range(8)]
if 'audio_generated' not in st.session_state:
    st.session_state.audio_generated = False

st.set_page_config(layout="wide")

# Compact styling
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-baseweb="select"] {border-radius: 4px !important;}
        .stSlider {height: 40px;}
        div[role="radiogroup"] > label {flex: 1;}
        .stDownloadButton button {width: 100% !important;}
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🎹 M42 - AI Music Workstation")
st.caption("Professional AI Music Generation with DAW Features")


# Model selection and loading
models = {
    "HuggingFace": "facebook/musicgen-small",
    "CarloLocal": "/home/gringo/Desktop/Msc/models/facebook/musicgen-small/"
}

# In the top header section
with st.container():
    col_model, col_space = st.columns([2, 8])
    with col_model:
        model_selection = st.selectbox("Model", options=list(models.keys()))
        

try:
    device = torch.device("cpu")
    model = MusicgenForConditionalGeneration.from_pretrained(models[st.session_state.get('model_selection', 'HuggingFace')]).to(device)
    processor = AutoProcessor.from_pretrained(models[st.session_state.get('model_selection', 'HuggingFace')])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Audio generation functions
def generate_music():
    full_prompt = f"{st.session_state.description} {', '.join(st.session_state.instruments)}. " \
                f"{st.session_state.scale_type} scale, {st.session_state.progression_type} progression, " \
                f"{st.session_state.mode_type} mood."
    
    if st.session_state.use_kodaly:
        full_prompt += f" {st.session_state.kodaly_rhythm} rhythm, {st.session_state.kodaly_form} structure, " \
                     f"emphasizing {st.session_state.kodaly_steps}% stepwise motion."

    inputs = processor(text=[full_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    
    try:
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=st.session_state.guidance_scale,
                max_new_tokens=int(st.session_state.desired_seconds * st.session_state.max_new_tokens_factor),
                temperature=st.session_state.temperature,
                top_k=st.session_state.top_k,
                top_p=st.session_state.top_p
            ).detach().cpu().numpy().ravel()

        audio_values = audio_values / np.max(np.abs(audio_values)) if np.max(np.abs(audio_values)) != 0 else audio_values
        
        # Apply audio processing in memory
        board = Pedalboard([])
        if st.session_state.apply_reverb:
            board.append(Reverb(room_size=st.session_state.reverb_room_size))
        if st.session_state.apply_compression:
            board.append(Compressor(ratio=st.session_state.compression_ratio))
        if st.session_state.apply_eq:
            board.append(PeakFilter(gain_db=st.session_state.eq_low_gain, cutoff_frequency_hz=100))
            board.append(PeakFilter(gain_db=st.session_state.eq_mid_gain, cutoff_frequency_hz=1000))
            board.append(PeakFilter(gain_db=st.session_state.eq_high_gain, cutoff_frequency_hz=5000))

        processed_audio = board(audio_values, 32000)
        
        # Save to in-memory buffer
        buffer = BytesIO()
        wavfile.write(buffer, 32000, (processed_audio * 32767.0).astype(np.int16))
        audio_bytes = buffer.getvalue()
        
        st.session_state.audio_generated = True
        st.session_state.tracks.append(("AI Track", audio_bytes))
        
    except Exception as e:
        st.error(f"Music generation error: {e}")
        st.session_state.audio_generated = False

def generate_drum_pattern(pattern, bpm, kit_type):
    midi = pretty_midi.PrettyMIDI()
    drum_program = pretty_midi.instrument_name_to_program('Synth Drum')
    instrument = pretty_midi.Instrument(program=drum_program)
    
    step_length = 60 / bpm / 4
    for i, drum in enumerate(pattern):
        for j, step in enumerate(drum):
            if step:
                start = j * step_length
                note = pretty_midi.Note(
                    velocity=100, pitch=36+i, 
                    start=start, end=start + step_length
                )
                instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    audio_data = midi.synthesize()
    audio = (audio_data * 32767).astype(np.int16)
    
    buffer = BytesIO()
    wavfile.write(buffer, 44100, audio)
    return buffer.getvalue()

def generate_sequence(pattern, bpm):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    step_length = 60 / bpm / 2
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    
    for i, row in enumerate(pattern):
        for j, step in enumerate(row):
            if step:
                start = j * step_length
                note = pretty_midi.Note(
                    velocity=100, pitch=notes[i], 
                    start=start, end=start + step_length
                )
                instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    audio_data = midi.synthesize()
    audio = (audio_data * 32767).astype(np.int16)
    
    buffer = BytesIO()
    wavfile.write(buffer, 44100, audio)
    return buffer.getvalue()

def mix_tracks(tracks):
    mixed = AudioSegment.silent(duration=1000)
    for _, audio_bytes in tracks:
        track = AudioSegment.from_wav(BytesIO(audio_bytes))
        mixed = mixed.overlay(track)
    
    buffer = BytesIO()
    mixed.export(buffer, format="wav")
    return buffer.getvalue()

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["Composer", "Drum Machine", "Sequencer", "DAW"])

with tab1:
    with st.container(border=True):
        st.subheader("🎼 Composition Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.music_type = st.selectbox("Genre", [
                "Classical", "Jazz", "Rock", "Electronic", "Cinematic",
                "Ambient", "Folk", "Experimental", "Hiphop", "Rap", 
                "Soul", "Kodály Folk"
            ])
            st.session_state.scale_type = st.selectbox("Scale", [
                "Major", "Minor", "Pentatonic (Kodály)", "Harmonic Minor",
                "Dorian", "Phrygian", "Lydian", "Mixolydian", "Blues"
            ])
            
        with col2:
            st.session_state.progression_type = st.selectbox("Progression", [
                "I-IV-V", "ii-V-I", "I-V-vi-IV", "I-vi-ii-V", "Custom...",
                "Modal", "Atonal", "Chromatic", "Kodály Folk (Stepwise)"
            ])
            st.session_state.mode_type = st.selectbox("Mood", [
                "Happy", "Melancholic", "Energetic", "Calm",
                "Mysterious", "Dramatic", "Playful", "Somber"
            ])

        st.session_state.instruments = st.multiselect("Instruments", [
            "Piano", "Strings", "Electric Guitar", "Drums",
            "Synthesizer", "Brass", "Woodwinds", "Harp", "Bass808"
        ], default=["Piano"], max_selections=4)
        
        st.session_state.desired_seconds = st.slider("Duration (seconds)", 10, 300, 30)

    with st.container(border=True):
        st.subheader("⚙️ Advanced Controls")
        
        gen_tab, proc_tab = st.tabs(["Generation", "Processing"])
        with gen_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Creativity", 0.1, 2.0, 0.8)
                st.session_state.top_k = st.slider("Top-K", 1, 100, 50)
            with col2:
                st.session_state.top_p = st.slider("Nucleus Sampling", 0.1, 1.0, 0.95)
                st.session_state.guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 8.0)
            st.session_state.max_new_tokens_factor = st.slider("Duration Factor", 10, 30, 20)

        with proc_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.apply_reverb = st.checkbox("Reverb", True)
                st.session_state.reverb_room_size = st.slider("Room Size", 0.0, 1.0, 0.5)
            with col2:
                st.session_state.apply_compression = st.checkbox("Compression", True)
                st.session_state.compression_ratio = st.slider("Ratio", 1.0, 10.0, 4.0)
            
            st.session_state.apply_eq = st.checkbox("Equalizer", True)
            eq_cols = st.columns(3)
            with eq_cols[0]: st.session_state.eq_low_gain = st.slider("Bass", -12, 12, 0)
            with eq_cols[1]: st.session_state.eq_mid_gain = st.slider("Mid", -12, 12, 0)
            with eq_cols[2]: st.session_state.eq_high_gain = st.slider("Treble", -12, 12, 0)

    with st.container(border=True):
        st.subheader("🚀 Generation Interface")
        st.session_state.description = st.text_area(
            "Prompt",
            f"A {st.session_state.music_type} composition in {st.session_state.scale_type} "
            f"scale with {st.session_state.progression_type} progression, "
            f"evoking a {st.session_state.mode_type} mood.",
            height=100
        )
        
        st.session_state.use_kodaly = st.checkbox("Enable Kodály Features", False)
        if st.session_state.use_kodaly:
            kod_cols = st.columns(2)
            with kod_cols[0]:
                st.session_state.kodaly_rhythm = st.selectbox("Rhythm", [
                    "ta (Quarter)", "ti-ti (Eighth)", 
                    "tika-tika (16th)", "Syncopation"
                ])
                st.session_state.kodaly_form = st.selectbox("Structure", [
                    "ABA", "AAB", "Call-Response"
                ])
            with kod_cols[1]:
                st.session_state.kodaly_steps = st.slider("Stepwise Motion", 50, 100, 80)
        
        if st.button("🎵 Generate Music", use_container_width=True, type="primary"):
            generate_music()

with tab2:
    with st.container(border=True):
        st.subheader("🥁 Drum Machine")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            drum_types = ["Kick", "Snare", "Hi-Hat", "Clap"]
            for i, drum in enumerate(drum_types):
                st.write(f"**{drum}**")
                cols = st.columns(16)
                for j in range(16):
                    st.session_state.drum_pattern[i][j] = cols[j].checkbox(
                        "", st.session_state.drum_pattern[i][j],
                        key=f"drum_{i}_{j}"
                    )
        
        with col2:
            bpm = st.slider("BPM", 60, 200, 120)
            kit_type = st.selectbox("Kit Type", ["808", "Acoustic", "Electronic"])
            if st.button("Generate Drum Pattern", use_container_width=True):
                drum_audio = generate_drum_pattern(
                    st.session_state.drum_pattern, 
                    bpm, 
                    kit_type
                )
                st.session_state.tracks.append(("Drums", drum_audio))
                st.audio(drum_audio, format="audio/wav")

with tab3:
    with st.container(border=True):
        st.subheader("🎹 Step Sequencer")
        
        notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
        for i, note in enumerate(notes):
            st.write(f"**{note}**")
            cols = st.columns(16)
            for j in range(16):
                st.session_state.sequencer_notes[i][j] = cols[j].checkbox(
                    "", st.session_state.sequencer_notes[i][j],
                    key=f"seq_{i}_{j}"
                )
        
        if st.button("Generate Sequence", use_container_width=True):
            seq_audio = generate_sequence(
                st.session_state.sequencer_notes, 
                st.session_state.get('bpm', 120)
            )
            st.session_state.tracks.append(("Sequence", seq_audio))
            st.audio(seq_audio, format="audio/wav")

with tab4:
    with st.container(border=True):
        st.subheader("🎚️ DAW Interface")
        
        for idx, (track_name, audio_bytes) in enumerate(st.session_state.tracks):
            with st.expander(f"Track {idx+1}: {track_name}", expanded=True):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    vol = st.slider(f"Vol {idx+1}", 0.0, 2.0, 1.0, key=f"vol_{idx}")
                    pan = st.slider(f"Pan {idx+1}", -1.0, 1.0, 0.0, key=f"pan_{idx}")
                with col2:
                    st.audio(audio_bytes, format="audio/wav")
                with col3:
                    st.download_button(
                        "Download Track",
                        data=audio_bytes,
                        file_name=f"{track_name}_{idx+1}.wav",
                        mime="audio/wav",
                        key=f"dl_{idx}",
                        use_container_width=True
                    )
        
        if st.button("🔀 Mix Down All Tracks", use_container_width=True):
            if st.session_state.tracks:
                final_mix = mix_tracks(st.session_state.tracks)
                st.audio(final_mix, format="audio/wav")
                st.download_button(
                    "Download Final Mix",
                    data=final_mix,
                    file_name="final_mix.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            else:
                st.warning("No tracks to mix!")
