🎹 M42 Music Workstation v0.4

AI-Powered Composition Suite with DAW Integration
"Where Kodály Tradition Meets Modern Music Production"

Python Version
License
🚀 Key Features
diff
Copy

+ Added Full Production Workflow
+ New In-Memory Audio Engine
+ Enhanced Kodály Integration



![Screenshot from 2025-02-15 01-01-21](https://github.com/user-attachments/assets/8db32520-629a-49c5-a852-11f24eb08b28)

![Screenshot from 2025-02-15 01-01-33](https://github.com/user-attachments/assets/56378748-6baf-4b8b-a397-0bf52df21282)

![Screenshot from 2025-02-15 01-01-43](https://github.com/user-attachments/assets/99469929-2179-4d68-900a-e904a39d2063)

![Screenshot from 2025-02-15 01-01-56](https://github.com/user-attachments/assets/8c1a3512-1792-4849-9447-f2fc72f4aab1)
Project Overview:




🎛️ Core Components
Module	Description
AI Composer	GPT-style music generation with Kodály constraints
Drum Machine	Pattern-based percussion sequencer (808/Acoustic/Electronic)
Step Sequencer	16-step MIDI composer with scale quantization
DAW Environment	Multi-track mixing/export with FX chain
📦 Installation
bash
Copy

# Clone repo
git clone https://github.com/yourusername/M42-Workstation.git

# Install dependencies
pip install -r requirements.txt

# Launch interface
streamlit run music_workstation.py

🛠️ What's Changed
New Features

    🥁 Integrated drum pattern designer

    🎹 Real-time MIDI sequencer grid

    💾 Memory-optimized audio handling (no temp files)

    📥 Track-specific WAV exports

Technical Improvements

    ⚡ 40% reduced memory usage

    🎚️ Professional-grade signal chain (Reverb/Compression/EQ)

    🔄 Dynamic model reloading

    📈 Improved cross-platform stability

🐛 Bug Fixes

    Fixed Streamlit config initialization error

    Resolved audio normalization artifacts

    Patched MIDI timing inconsistencies

📋 Requirements
plaintext
Copy

Minimum:
- CPU: Intel i5 / Ryzen 5 (4 cores)
- RAM: 8GB 
- Storage: 2GB (5GB for local models)

Recommended:
- CPU: Intel i7 / Ryzen 7 (8 cores)  
- RAM: 16GB  
- Storage: SSD with 10GB free space  

📜 Full Changelog

See CHANGELOG.md for detailed version history.

Contribution Welcome!
🐞 Found an issue? Open a ticket
💡 Have a feature request? Start a discussion

This format uses GitHub-flavored markdown with:

    Shield.io badges for version tracking

    Diff-style feature highlights

    Responsive tables

    Code block installation instructions

    Clear issue tracking links

    Mobile-friendly spacing

Would you like me to add specific contribution guidelines or development documentation sections?
New chat
AI-generated, for reference only






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
