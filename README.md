# Kokoro TTS Local

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![TTS](https://img.shields.io/badge/TTS-Text--to--Speech-orange.svg)](https://github.com/solveditnpc/kokoro-tts-local)

A powerful, offline Text-to-Speech (TTS) solution based on the Kokoro-82M model, featuring 44 high-quality voices across multiple languages and accents. This local implementation provides fast, reliable text-to-speech conversion with support for multiple output formats (WAV, MP3, AAC) and real-time generation progress display.

## 🌟 Key Features

- 🎙️ 44 high-quality voices across American English, British English, and other languages
- 💻 Completely offline operation - no internet needed after initial setup
- 📚 Support for PDF and TXT file input
- 🎵 Multiple output formats (WAV, MP3, AAC)
- ⚡ Real-time generation with progress display
- 🎛️ Adjustable speech speed (0.5x to 2.0x)
- 📊 Automatic text chunking for optimal processing
- 🎯 Easy-to-use interactive CLI interface

## Creating Custom Voices and creating audio books
   
   creating custom voices
   ```
   python custom_interpolation.py
   ```

   creating audio books
   ```
   python audio_book.py
   ```
   Note - you need to install the prerequisites and follow the installation steps before running the above commands

## Installing Prerequisites

Before installing Kokoro TTS Local, ensure you have the following prerequisites installed from the below guide:

- Python 3.10.0 or higher
- FFmpeg (for MP3/AAC conversion)
- CUDA-compatible GPU (optional, for faster generation)
- Git (for version control and package management)

### Installing Git

#### Windows
1. Download Git installer:
   ```cmd
   winget install --id Git.Git -e --source winget
   ```
   Alternatively, download from [Git for Windows](https://gitforwindows.org/)

2. Verify installation:
   ```cmd
   git --version
   ```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git

# Fedora
sudo dnf install git

# Arch Linux
sudo pacman -S git

# Verify installation
git --version
```

#### macOS
```bash
# Using Homebrew
brew install git

# Verify installation
git --version
```

### Installing FFmpeg

#### Windows
1. ```cmd
   iex (irm ffmpeg.tc.ht)
   ```
2. Verify installation by opening a new Command Prompt:
   ```cmd
   ffmpeg -version
   ```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg
```

#### macOS
```bash
# Using Homebrew
brew install ffmpeg
```

### Installing CUDA Drivers

#### Windows
1. Check your GPU compatibility:
   - Open Command Prompt and run: `dxdiag`
   - Go to the "Display" tab
   - Note your GPU model

2. Download CUDA Toolkit:
   - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select Windows and your version
   - Choose "exe (network)" installer
   - Download and run the installer

3. Installation steps:
   - Run the downloaded installer
   - Choose "Express Installation"
   - Wait for the installation to complete
   - Restart your computer

4. Verify installation:
   ```cmd
   nvidia-smi
   nvcc --version
   ```

#### Linux
1. Check your GPU compatibility:
   ```bash
   lspci | grep -i nvidia
   ```

2. Remove old NVIDIA drivers (if any):
   ```bash
   sudo apt-get purge nvidia*
   ```

3. Add NVIDIA package repositories:
   ```bash
   # Ubuntu 22.04 LTS(install drivers based on your distro release, 22.04 drivers are no longer compatible with the 24.01 or 24.02 version)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
   ```

4. Install CUDA drivers:
   ```bash
   sudo apt-get update
   sudo apt-get -y install cuda-drivers
   ```

5. Install CUDA Toolkit:
   ```bash
   sudo apt-get install cuda
   ```

6. Add CUDA to PATH:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
   source ~/.bashrc
   ```

7. Verify installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

Note: For macOS, CUDA is not supported natively. The model will run on CPU only.

## Installation

### Windows Installation

1. **Install Python 3.10.0**
   - Download the installer from [Python's official website](https://www.python.org/downloads/release/python-3100/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install espeak-ng**
   - Download the latest release from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
   - Run the installer and follow the prompts
   - Add espeak-ng to your system PATH if not done automatically

3. Clone the repository:
   ```cmd
   git clone https://github.com/solveditnpc/kokoro-tts-local.git
   cd kokoro-tts-local
   ```

4. Create and activate a virtual environment:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

5. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

### Linux Installation
0. **install espeak**
   ```bash
   # Install espeak-ng
   sudo apt-get install espeak-ng
   ```

1. **Install Dependencies**
   ```bash
   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
   libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
   liblzma-dev python-openssl git
   ```

2. **Install pyenv**
   ```bash
   # Install pyenv
   curl https://pyenv.run | bash
   
   # Add to ~/.bashrc
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   
   # Reload shell
   exec "$SHELL"
   ```

3. **Install Python 3.10.0**
   ```bash
   # Install Python 3.10.0
   pyenv install 3.10.0
   
   # Clone repository
   git clone https://github.com/solveditnpc/kokoro-tts-local.git
   cd kokoro-tts-local
   
   # Set local Python version
   pyenv local 3.10.0
   ```

4. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### macOS Installation

1. **Install Dependencies**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install system dependencies
   brew install openssl readline sqlite3 xz zlib tcl-tk git
   
   # Install espeak-ng
   brew install espeak-ng
   ```

2. **Install pyenv**
   ```bash
   # Install pyenv
   brew install pyenv
   
   # Add to ~/.zshrc (or ~/.bashrc if using bash)
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   
   # Reload shell
   exec "$SHELL"
   ```

3. **Install Python 3.10.0**
   ```bash
   # Install Python 3.10.0
   pyenv install 3.10.0
   
   # Clone repository
   git clone https://github.com/solveditnpc/kokoro-tts-local.git
   cd kokoro-tts-local
   
   # Set local Python version
   pyenv local 3.10.0
   ```

4. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Available Voices

The system includes 44 different voices across various categories:

### American English Voices
- Female (af_*):
  - af_alloy: Alloy - Clear and professional
  - af_aoede: Aoede - Smooth and melodic
  - af_bella: Bella - Warm and friendly
  - af_jessica: Jessica - Natural and engaging
  - af_kore: Kore - Bright and energetic
  - af_nicole: Nicole - Professional and articulate
  - af_nova: Nova - Modern and dynamic
  - af_river: River - Soft and flowing
  - af_sarah: Sarah - Casual and approachable
  - af_sky: Sky - Light and airy

- Male (am_*):
  - am_adam: Adam - Strong and confident
  - am_echo: Echo - Resonant and clear
  - am_eric: Eric - Professional and authoritative
  - am_fenrir: Fenrir - Deep and powerful
  - am_liam: Liam - Friendly and conversational
  - am_michael: Michael - Warm and trustworthy
  - am_onyx: Onyx - Rich and sophisticated
  - am_puck: Puck - Playful and energetic

### British English Voices
- Female (bf_*):
  - bf_alice: Alice - Refined and elegant
  - bf_emma: Emma - Warm and professional
  - bf_isabella: Isabella - Sophisticated and clear
  - bf_lily: Lily - Sweet and gentle

- Male (bm_*):
  - bm_daniel: Daniel - Polished and professional
  - bm_fable: Fable - Storytelling and engaging
  - bm_george: George - Classic British accent
  - bm_lewis: Lewis - Modern British accent

### Special Voices
- French Female (ff_*):
  - ff_siwis: Siwis - French accent

- High-pitched Voices:
  - Female (hf_*):
    - hf_alpha: Alpha - Higher female pitch
    - hf_beta: Beta - Alternative high female pitch
  - Male (hm_*):
    - hm_omega: Omega - Higher male pitch
    - hm_psi: Psi - Alternative high male pitch

## Project Structure

```
.
├── .cache/                 # Cache directory for downloaded models
│   └── huggingface/       # Hugging Face model cache
├── .git/                   # Git repository data
├── .gitignore             # Git ignore rules
├── __pycache__/           # Python cache files
├── voices/                # Voice model files (downloaded on demand)
│   └── *.pt              # Individual voice files
├── venv/                  # Python virtual environment
├── outputs/               # Generated audio files directory
├── LICENSE                # Apache 2.0 License file
├── README.md             # Project documentation
├── models.py             # Core TTS model implementation
├── gradio_interface.py   # Web interface implementation
├── config.json           # Model configuration file
├── requirements.txt      # Python dependencies
└── tts_demo.py          # CLI implementation
```

## Model Information

The project uses the latest Kokoro model from Hugging Face:
- Repository: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Model file: `kokoro-v1_0.pth` (downloaded automatically)
- Sample rate: 24kHz
- Voice files: Located in the `voices/` directory (downloaded automatically)
- Available voices: 44 voices across multiple categories
- Languages: American English ('a'), British English ('b')
- Model size: 82M parameters

## Troubleshooting

Common issues and solutions:

1. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face is accessible
   - Verify sufficient disk space
   - Try clearing the `.cache/huggingface` directory

2. **CUDA/GPU Issues**
   - Verify CUDA installation with `nvidia-smi`
   - Update GPU drivers
   - Check PyTorch CUDA compatibility
   - Fall back to CPU if needed

3. **Audio Output Issues**
   - Check system audio settings
   - Verify output directory permissions
   - Install FFmpeg for MP3/AAC support
   - Try different output formats

4. **Voice File Issues**
   - Delete and let system redownload voice files
   - Check `voices/` directory permissions
   - Verify voice file integrity
   - Try using a different voice

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Helping with documentation
4. Testing different voices and reporting issues
5. Suggesting new features or optimizations
6. Testing on different platforms and reporting results

## License

Apache 2.0 - See LICENSE file for details 