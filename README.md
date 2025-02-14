# M42 V 0.1 Beta
Musicgen based AI Music generator

M42 - V0.1





![Screenshot from 2025-02-14 06-09-46](https://github.com/user-attachments/assets/7575ce4f-b006-4d81-9f04-f0b60e91b227)










Project Overview:
M42 - V0.1 is a professional-grade AI music generation application built using Streamlit. It leverages Meta's MusicGen model for text-based music generation and includes advanced controls for fine-tuning musical parameters and audio post-processing.

Features:

AI-powered music generation using text descriptions.

Adjustable musical composition parameters (scale, progression, mood, etc.).

Advanced sampling controls (temperature, top-k, top-p, guidance scale).

Built-in audio post-processing (filters, compression, reverb, delay, distortion, etc.).

Local and Hugging Face-hosted model support.

Installation Instructions:

Clone the repository:

git clone https://github.com/your-repo/M42.git
cd M42

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

Install dependencies:

pip install -r requirements.txt

Model Links:

Hugging Face Hosted Model: facebook/musicgen-small

Local Model Path Example: /home/gringo/Desktop/Msc/models/facebook/musicgen-small/

Required files for local models:

config.json

pytorch_model.bin

preprocessor_config.json

Requirements.txt:

streamlit==1.28.0
transformers==4.45.1
torch
numpy
scipy
pedalboard

License:
M42 - V0.1 is provided for personal and research use only. Commercial use is strictly prohibited without prior written approval from the author. For commercial licensing inquiries, please contact [your email or contact method].


