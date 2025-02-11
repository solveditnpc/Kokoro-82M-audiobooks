"""Models module for Kokoro TTS Local"""
from typing import Optional, Tuple, List
import torch
from kokoro import KPipeline
import os
import json
import codecs
from pathlib import Path
import numpy as np
import shutil

# Set environment variables for proper encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# List of available voice files
VOICE_FILES = [
    # American Female voices
    "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
    "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt",
    "af_sarah.pt", "af_sky.pt",
    # American Male voices
    "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt",
    "am_liam.pt", "am_michael.pt", "am_onyx.pt", "am_puck.pt",
    "am_santa.pt",
    # British Female voices
    "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",
    # British Male voices
    "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",
    # Special voices
    "el_dora.pt", "em_alex.pt", "em_santa.pt",
    "ff_siwis.pt",
    "hf_alpha.pt", "hf_beta.pt",
    "hm_omega.pt", "hm_psi.pt",
    "jf_sara.pt", "jm_nicola.pt",
    "jf_alpha.pt", "jf_gongtsuene.pt", "jf_nezumi.pt", "jf_tebukuro.pt",
    "jm_kumo.pt",
    "pf_dora.pt", "pm_alex.pt", "pm_santa.pt",
    "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoqiao.pt", "zf_xiaoyi.pt"
]

# Patch KPipeline's load_voice method to use weights_only=False
original_load_voice = KPipeline.load_voice

def patched_load_voice(self, voice_path):
    """Load voice model with weights_only=False for compatibility"""
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    voice_name = Path(voice_path).stem
    voice_model = torch.load(voice_path, weights_only=False)
    if voice_model is None:
        raise ValueError(f"Failed to load voice model from {voice_path}")
    # Ensure device is set
    if not hasattr(self, 'device'):
        self.device = 'cpu'
    # Move model to device and store in voices dictionary
    self.voices[voice_name] = voice_model.to(self.device)
    return self.voices[voice_name]

KPipeline.load_voice = patched_load_voice

def patch_json_load():
    """Patch json.load to handle UTF-8 encoded files with special characters"""
    original_load = json.load
    
    def custom_load(fp, *args, **kwargs):
        try:
            # Try reading with UTF-8 encoding
            if hasattr(fp, 'buffer'):
                content = fp.buffer.read().decode('utf-8')
            else:
                content = fp.read()
            return json.loads(content)
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig for files with BOM
            fp.seek(0)
            content = fp.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig', errors='replace')
            return json.loads(content)
    
    json.load = custom_load

def load_config(config_path: str) -> dict:
    """Load configuration file with proper encoding handling"""
    try:
        with codecs.open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if regular utf-8 fails
        with codecs.open(config_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

# Initialize espeak-ng
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer import phonemize
    import espeakng_loader
    
    # Make library available first
    library_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    espeakng_loader.make_library_available()
    
    # Set up espeak-ng paths
    EspeakWrapper.library_path = library_path
    EspeakWrapper.data_path = data_path
    
    # Verify espeak-ng is working
    try:
        test_phonemes = phonemize('test', language='en-us')
        if not test_phonemes:
            raise Exception("Phonemization returned empty result")
    except Exception as e:
        print(f"Warning: espeak-ng test failed: {e}")
        print("Some functionality may be limited")

except ImportError as e:
    print(f"Warning: Required packages not found: {e}")
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call(["pip", "install", "espeakng-loader", "phonemizer-fork"])
    
    # Try again after installation
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer import phonemize
    import espeakng_loader
    
    library_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    espeakng_loader.make_library_available()
    EspeakWrapper.library_path = library_path
    EspeakWrapper.data_path = data_path

# Initialize pipeline globally
_pipeline = None

def download_voice_files():
    """Download voice files from Hugging Face."""
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    downloaded_voices = []
    
    print("\nDownloading voice files...")
    for voice_file in VOICE_FILES:
        try:
            # Full path where the voice file should be
            voice_path = voices_dir / voice_file
            
            if not voice_path.exists():
                print(f"Downloading {voice_file}...")
                # Download to a temporary location first
                temp_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename=f"voices/{voice_file}",
                    local_dir="temp_voices",
                    force_download=True
                )
                
                # Move the file to the correct location
                os.makedirs(os.path.dirname(voice_path), exist_ok=True)
                shutil.move(temp_path, voice_path)
                downloaded_voices.append(voice_file)
                print(f"Successfully downloaded {voice_file}")
            else:
                print(f"Voice file {voice_file} already exists")
                downloaded_voices.append(voice_file)
        except Exception as e:
            print(f"Warning: Failed to download {voice_file}: {e}")
            continue
    
    # Clean up temporary directory
    if os.path.exists("temp_voices"):
        shutil.rmtree("temp_voices")
    
    if not downloaded_voices:
        print("Warning: No voice files could be downloaded. Please check your internet connection.")
    else:
        print(f"Successfully processed {len(downloaded_voices)} voice files")
    
    return downloaded_voices

def build_model(model_path: str, device: str) -> KPipeline:
    """Build and return the Kokoro pipeline with proper encoding configuration"""
    global _pipeline
    if _pipeline is None:
        try:
            # Patch json loading before initializing pipeline
            patch_json_load()
            
            # Download model if it doesn't exist
            if model_path is None:
                model_path = 'kokoro-v1_0.pth'
            
            if not os.path.exists(model_path):
                print(f"Downloading model file {model_path}...")
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename="kokoro-v1_0.pth",
                    local_dir=".",
                    force_download=True
                )
                print(f"Model downloaded to {model_path}")
            
            # Download config if it doesn't exist
            config_path = "config.json"
            if not os.path.exists(config_path):
                print("Downloading config file...")
                config_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename="config.json",
                    local_dir=".",
                    force_download=True
                )
                print(f"Config downloaded to {config_path}")
            
            # Download voice files
            downloaded_voices = download_voice_files()
            
            if not downloaded_voices:
                print("Error: No voice files available. Cannot proceed.")
                raise ValueError("No voice files available")
            
            # Initialize pipeline with American English by default
            _pipeline = KPipeline(lang_code='a')
            if _pipeline is None:
                raise ValueError("Failed to initialize KPipeline - pipeline is None")
                
            # Store device parameter for reference in other operations
            _pipeline.device = device
            
            # Initialize voices dictionary if it doesn't exist
            if not hasattr(_pipeline, 'voices'):
                _pipeline.voices = {}
            
            # Try to load the first available voice
            for voice_file in downloaded_voices:
                voice_path = f"voices/{voice_file}"
                if os.path.exists(voice_path):
                    try:
                        _pipeline.load_voice(voice_path)
                        print(f"Successfully loaded voice: {voice_file}")
                        break  # Successfully loaded a voice
                    except Exception as e:
                        print(f"Warning: Failed to load voice {voice_file}: {e}")
                        continue
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise
    return _pipeline

def list_available_voices() -> List[str]:
    """List all available voice models"""
    voices_dir = Path("voices")
    
    # Create voices directory if it doesn't exist
    if not voices_dir.exists():
        print(f"Creating voices directory at {voices_dir.absolute()}")
        voices_dir.mkdir(exist_ok=True)
        return []
    
    # Get all .pt files in the voices directory
    voice_files = list(voices_dir.glob("*.pt"))
    
    # If no voice files found in voices directory
    if not voice_files:
        print(f"No voice files found in {voices_dir.absolute()}")
        # Try to find voice files in the root directory's voices folder
        root_voices = list(Path(".").glob("voices/*.pt"))
        if root_voices:
            print("Found voice files in root voices directory, moving them...")
            for voice_file in root_voices:
                target_path = voices_dir / voice_file.name
                if not target_path.exists():
                    shutil.move(str(voice_file), str(target_path))
            # Recheck voices directory
            voice_files = list(voices_dir.glob("*.pt"))
    
    if not voice_files:
        print("No voice files found. Please run the application again to download voices.")
        return []
    
    return [f.stem for f in voice_files]

def load_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load a voice model"""
    pipeline = build_model(None, device)
    # Format voice path correctly - strip .pt if it was included
    voice_name = voice_name.replace('.pt', '')
    voice_path = f"voices/{voice_name}.pt"
    if not os.path.exists(voice_path):
        raise ValueError(f"Voice file not found: {voice_path}")
    return pipeline.load_voice(voice_path)

def generate_speech(
    model: KPipeline,
    text: str,
    voice: str,
    lang: str = 'a',
    device: str = 'cpu',
    speed: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Generate speech using the Kokoro pipeline
    
    Args:
        model: KPipeline instance
        text: Text to synthesize
        voice: Voice name (e.g. 'af_bella')
        lang: Language code ('a' for American English, 'b' for British English)
        device: Device to use ('cuda' or 'cpu')
        speed: Speech speed multiplier (default: 1.0)
        
    Returns:
        Tuple of (audio tensor, phonemes string) or (None, None) on error
    """
    try:
        if model is None:
            raise ValueError("Model is None - pipeline not properly initialized")
            
        # Initialize voices dictionary if it doesn't exist
        if not hasattr(model, 'voices'):
            model.voices = {}
            
        # Ensure device is set
        if not hasattr(model, 'device'):
            model.device = device
            
        # Format voice path and ensure voice is loaded
        voice_name = voice.replace('.pt', '')
        voice_path = f"voices/{voice_name}.pt"
        if not os.path.exists(voice_path):
            raise ValueError(f"Voice file not found: {voice_path}")
            
        # Ensure voice is loaded before generating
        if voice_name not in model.voices:
            print(f"Loading voice {voice_name}...")
            model.load_voice(voice_path)
            
        if voice_name not in model.voices:
            raise ValueError(f"Failed to load voice {voice_name}")
            
        # Generate speech with the new API
        print(f"Generating speech with device: {model.device}")
        generator = model(
            text, 
            voice=voice_path,
            speed=speed,
            split_pattern=r'\n+'
        )
        
        # Get first generated segment and convert numpy array to tensor if needed
        for gs, ps, audio in generator:
            if audio is not None:
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                return audio, ps
            
        return None, None
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None, None