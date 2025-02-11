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

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

def print_menu():
    """Print the main menu options."""
    print("\n=== Kokoro TTS Multi-Line Menu ===")
    print("1. List available voices")
    print("2. Generate speech from text")
    print("3. Generate speech from file")
    print("4. Exit")
    return input("Select an option (1-4): ").strip()

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

def get_file_input() -> List[str]:
    """Get text input from a file (supports both .txt and .pdf files)."""
    while True:
        file_path = input("\nEnter the path to your text or PDF file: ").strip()
        if os.path.exists(file_path):
            try:
                # Check file extension
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    return extract_text_from_pdf(file_path)
                elif file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    return lines if lines else [DEFAULT_TEXT]
                else:
                    print("Unsupported file format. Please use .txt or .pdf files.")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print("File not found. Please try again.")

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
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"outputs/output_{timestamp}.wav")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {idx}/{len(chunks)}: '{chunk}'")
        
        chunk_audio = []
        generator = model(chunk, voice=f"voices/{voice}.pt", speed=speed)
        
        with tqdm(desc=f"Generating speech for chunk {idx}") as pbar:
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    chunk_audio.append(audio)
                    print(f"\nGenerated segment: {gs}")
                    print(f"Phonemes: {ps}")
                    pbar.update(1)
        
        if chunk_audio:
            # Combine audio for this chunk
            chunk_combined = torch.cat(chunk_audio, dim=0)
            
            # Add appropriate pause based on punctuation
            last_char = chunk.strip()[-1] if chunk.strip() else ''
            if last_char in '.!?':
                pause_length = int(SAMPLE_RATE * 0.25)  # Longer pause for sentence endings
            elif last_char in ',;:':
                pause_length = int(SAMPLE_RATE * 0.15)  # Medium pause for clause breaks
            else:
                pause_length = int(SAMPLE_RATE * 0.07)  # Short pause for natural breathing
            
            pause = torch.zeros(pause_length)
            all_audio_segments.extend([chunk_combined, pause])
        else:
            print(f"Error: Failed to generate audio for chunk {idx}")
    
    if all_audio_segments:
        # Combine all segments into one audio file
        final_audio = torch.cat(all_audio_segments, dim=0)
        sf.write(output_path, final_audio.numpy(), SAMPLE_RATE)
        print(f"\nAll audio combined and saved to {output_path.absolute()}")
    else:
        print("Error: Failed to generate any audio")

def main() -> None:
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize model directly without verification
        model = build_model(DEFAULT_MODEL_PATH, device)
        
        while True:
            choice = print_menu()
            
            if choice == "1":
                # List voices
                voices = list_available_voices()
                print("\nAvailable voices:")
                for voice in voices:
                    print(f"- {voice}")
                    
            elif choice in ["2", "3"]:
                # Generate speech
                voices = list_available_voices()
                if not voices:
                    print("No voices found! Please check the voices directory.")
                    continue
                
                # Get user inputs
                voice = select_voice(voices)
                text_lines = get_file_input() if choice == "3" else get_text_input()
                speed = get_speed()
                
                # Generate audio for all lines
                generate_audio(model, text_lines, voice, speed)
                
            elif choice == "4":
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
