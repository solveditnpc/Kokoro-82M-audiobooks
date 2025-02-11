import torch
from typing import Optional, Tuple, List
from models import build_model, generate_speech, list_available_voices
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np
import os
import PyPDF2
import datetime

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = 'kokoro-v1_0.pth'
DEFAULT_OUTPUT_FILE = 'outputs/output.wav'
DEFAULT_LANGUAGE = 'a'  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0

# Create outputs and input directories if they don't exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('input', exist_ok=True)

def print_menu():
    """Print the main menu options."""
    print("\n=== Kokoro TTS Multi-Line Menu ===")
    print("1. Generate speech from text")
    print("2. Generate speech from PDF file or TXT file")
    print("3. Exit")
    return input("Select an option (1-3): ").strip()

def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    
    while True:
        try:
            choice = input("\nSelect a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_text_input() -> List[str]:
    """Get multi-line text input from user."""
    print("\nEnter the text you want to convert to speech (enter an empty line to finish):")
    lines = []
    while True:
        line = input("> ").strip()
        if not line and lines:  # Empty line and we have content
            break
        elif not line and not lines:  # Empty line but no content yet
            return [DEFAULT_TEXT]
        lines.append(line)
    return lines

def extract_text_from_pdf(file_path: str) -> List[str]:
    """Extract text from a PDF file and return it as a list of lines."""
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Ask user for page range
            print(f"\nThe PDF has {total_pages} pages.")
            while True:
                try:
                    start_page = int(input(f"Enter start page (1-{total_pages}): ").strip())
                    end_page = int(input(f"Enter end page (1-{total_pages}): ").strip())
                    
                    if 1 <= start_page <= end_page <= total_pages:
                        break
                    print(f"Please enter valid page numbers between 1 and {total_pages}")
                except ValueError:
                    print("Please enter valid numbers")
            
            # Extract text from selected pages
            text_lines = []
            for page_num in range(start_page - 1, end_page):
                text = pdf_reader.pages[page_num].extract_text()
                if text:
                    # Split text into lines and filter out empty lines
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    text_lines.extend(lines)
            
            return text_lines if text_lines else [DEFAULT_TEXT]
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return [DEFAULT_TEXT]

def find_input_files() -> List[str]:
    """Find all PDF and TXT files in the input directory."""
    input_dir = Path('input')
    files = []
    for ext in ['.pdf', '.txt']:
        files.extend(list(input_dir.glob(f'*{ext}')))
    return [str(f) for f in files]

def select_input_file(files: List[str]) -> str:
    """Let user select a file from the list of available files."""
    print("\nAvailable files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    while True:
        try:
            choice = input("\nSelect a file number: ").strip()
            choice = int(choice)
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_file_input() -> List[str]:
    """Get text input from a file (supports both .txt and .pdf files)."""
    # Find all input files
    input_files = find_input_files()
    
    if not input_files:
        print("\nNo PDF or TXT files found in the input folder.")
        print("Please place your files in the 'input' folder and try again.")
        return [DEFAULT_TEXT]
    
    # If only one file, use it directly
    if len(input_files) == 1:
        file_path = input_files[0]
        print(f"\nUsing file: {os.path.basename(file_path)}")
    else:
        file_path = select_input_file(input_files)
    
    try:
        # Check file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines if lines else [DEFAULT_TEXT]
    except Exception as e:
        print(f"Error reading file: {e}")
        return [DEFAULT_TEXT]

def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input("\nEnter speech speed (0.5-2.0, default 1.0): ").strip()
            if not speed:
                return 1.0
            speed = float(speed)
            if 0.5 <= speed <= 2.0:
                return speed
            print("Speed must be between 0.5 and 2.0")
        except ValueError:
            print("Please enter a valid number.")

def get_audio_format() -> Tuple[str, str]:
    """Get desired audio format from user."""
    print("\nAvailable audio formats:")
    formats = {
        "1": ("wav", "WAV - Highest quality, larger file size"),
        "2": ("mp3", "MP3 - Good quality, smaller file size"),
        "3": ("aac", "AAC - Good quality, smallest file size")
    }
    
    for key, (fmt, desc) in formats.items():
        print(f"{key}. {fmt.upper()} - {desc}")
    
    while True:
        choice = input("\nSelect audio format (1-3, default: wav): ").strip()
        if not choice:
            return "wav", ".wav"
        if choice in formats:
            fmt = formats[choice][0]
            return fmt, f".{fmt}"
        print("Invalid choice. Please try again.")

def split_text_into_chunks(text: str) -> List[str]:
    """
    Split text into natural chunks based on punctuation and length.
    Ensures words aren't split in the middle and maintains sentence structure.
    """
    # Define punctuation marks that indicate natural breaks
    major_breaks = '.!?'
    minor_breaks = ',;:'
    max_chunk_length = 150  # Maximum characters per chunk
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into words while preserving punctuation
    words = text.replace('\n', ' ').split(' ')
    
    for word in words:
        word = word.strip()
        if not word:
            continue
            
        # Check if adding this word would exceed max length
        if current_length + len(word) + 1 > max_chunk_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(word)
        current_length += len(word) + 1
        
        # Check for natural breaks
        if word and word[-1] in major_breaks:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        elif word and word[-1] in minor_breaks and current_length > max_chunk_length/2:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_audio(model, text_lines: List[str], voice: str, speed: float) -> None:
    """Generate audio for multiple lines of text and combine into a single file."""
    all_audio_segments = []
    
    # Join all lines with appropriate spacing
    full_text = ' '.join(text_lines)
    
    # Split text into natural chunks
    chunks = split_text_into_chunks(full_text)
    
    # Get desired audio format
    format, extension = get_audio_format()
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"outputs/output_{timestamp}{extension}")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {idx}/{len(chunks)}: '{chunk}'")
        
        chunk_audio = []
        generator = model(chunk, voice=f"voices/{voice}.pt", speed=speed)
        
        with tqdm(desc="Generating") as pbar:
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    chunk_audio.append(audio)
                    pbar.update(1)
        
        if chunk_audio:
            # Combine audio for this chunk
            chunk_combined = torch.cat(chunk_audio, dim=0)
            all_audio_segments.append(chunk_combined.numpy())
            
            # Add silence between chunks
            silence = np.zeros(int(SAMPLE_RATE * 0.5))  # 0.5s silence
            all_audio_segments.append(silence)
    
    if all_audio_segments:
        # Combine all audio segments
        audio_array = np.concatenate(all_audio_segments)
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Save the audio file in the desired format
        if format == "wav":
            sf.write(output_path, audio_array, SAMPLE_RATE)
        else:
            # Save as WAV first
            temp_wav = output_path.with_suffix('.wav')
            sf.write(temp_wav, audio_array, SAMPLE_RATE)
            
            # Convert to desired format using FFmpeg
            try:
                if format == "mp3":
                    os.system(f'ffmpeg -i "{temp_wav}" -codec:a libmp3lame -qscale:a 2 "{output_path}"')
                elif format == "aac":
                    os.system(f'ffmpeg -i "{temp_wav}" -c:a aac -b:a 192k "{output_path}"')
                
                # Remove temporary WAV file
                temp_wav.unlink()
                print(f"\nAudio saved as: {output_path}")
            except Exception as e:
                print(f"Error converting to {format.upper()}: {e}")
                print(f"WAV file saved as: {temp_wav}")
                return
    else:
        print("No audio was generated. Please check if the input text is not empty.")

def main() -> None:
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize model directly without verification
        model = build_model(DEFAULT_MODEL_PATH, device)
        
        voices = list_available_voices()
        
        while True:
            choice = print_menu()
            
            if choice == "1":
                # Generate speech from text
                text_lines = get_text_input()
                voice = select_voice(voices)
                speed = get_speed()
                generate_audio(model, text_lines, voice, speed)
            
            elif choice == "2":
                # Generate speech from PDF file or TXT file
                text_lines = get_file_input()
                voice = select_voice(voices)
                speed = get_speed()
                generate_audio(model, text_lines, voice, speed)
            
            elif choice == "3":
                print("\nGoodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
