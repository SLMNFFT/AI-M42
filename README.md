# AI - M42 V 0.1 Beta
        AI Music generator

        M42 - V0.1





![Screenshot from 2025-02-14 06-09-46](https://github.com/user-attachments/assets/7575ce4f-b006-4d81-9f04-f0b60e91b227)










Project Overview:

        M42 - V0.1 is a professional-grade AI music generation application built using Streamlit. It leverages Meta's MusicGen model for text-based music generation and includes advanced controls for fine-tuning musical parameters and                audio post-processing.

Features:

        - AI-powered music generation using text descriptions.
        - Adjustable musical composition parameters (scale, progression, mood, etc.).
        - Advanced sampling controls (temperature, top-k, top-p, guidance scale).
        - Built-in audio post-processing (filters, compression, reverb, delay, distortion, etc.).
        - Local and Hugging Face-hosted model support.

Installation Instructions:

Description of Key Folders:

            /models/facebook/musicgen-small: Directory where the local model files (e.g., config.json, pytorch_model.bin,                   preprocessor_config.json) are stored. This is used if you are running the local version of the model.
            /assets: Folder to store any generated or processed audio files, such as the output of the music generation process.
            /app.py: This is the main file containing the Streamlit app code you provided.
            /requirements.txt: (optional) A file to list the Python dependencies for the project, e.g., torch, pedalboard,                  transformers, etc.
            /README.md: (optional) A file to explain the purpose of the project, usage instructions, and setup details.

        /M42_V0.1
                ├── /models

                │   └── /facebook

                │       └── /musicgen-small / other Models

                ├── /assets

                │   └── /processed_output.wav

                │   └── /musicgen_output.wav

                ├── /app.py  # Main Streamlit app (the script you provided)

                ├── /requirements.txt  # If needed to list dependencies like torch, pedalboard, etc.

                └── /README.md  # Documentation about the project (if needed)

Clone the repository:

                git clone https://github.com/SLMNFFT/M42.git
                cd M42

Create a virtual environment (optional but recommended):

                python -m venv venv
                source venv/bin/activate  # On macOS/Linux
                venv\Scripts\activate     # On Windows

Install dependencies:

                pip install -r requirements.txt

Model Links:
Hugging Face Hosted Model: facebook/musicgen-small
Local Model Path Example: /home/user/Desktop/folder/models/facebook/musicgen-small/


You can download the models from : [HuggingFace](https://huggingface.co/models?other=musicgen)

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

                M42 - V0.1 is provided for personal and research use only. Commercial use is strictly prohibited without prior                  written approval from the author. For commercial licensing inquiries, please contact [slimfatti@gmail.com].
