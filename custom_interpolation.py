import torch
from typing import Optional, List
from models import list_available_voices
import os

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def select_voice(voices: List[str], prompt: str = "\nSelect a voice number: ") -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    
    while True:
        try:
            choice = input(prompt).strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_interpolation_ratio() -> float:
    """Get interpolation ratio from user."""
    while True:
        try:
            ratio = input("\nEnter interpolation ratio (0.0-1.0, where 0.0 is fully first voice, 1.0 is fully second voice): ").strip()
            ratio = float(ratio)
            if 0.0 <= ratio <= 1.0:
                return ratio
            print("Ratio must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number.")

def get_custom_voice_name() -> str:
    """Get the name for the custom voice from user."""
    while True:
        name = input("\nEnter a name for your custom voice (e.g., 'custom_mix1'): ").strip()
        if name:
            # Remove .pt extension if user added it
            if name.endswith('.pt'):
                name = name[:-3]
            return name
        print("Please enter a valid name.")

def interpolate_voices(voice1_path: str, voice2_path: str, ratio: float) -> Optional[torch.Tensor]:
    """Interpolate between two voice embeddings."""
    try:
        # Add .pt extension if not present
        voice1_file = voice1_path if voice1_path.endswith('.pt') else f"{voice1_path}.pt"
        voice2_file = voice2_path if voice2_path.endswith('.pt') else f"{voice2_path}.pt"
        
        # Load voices
        try:
            voice1 = torch.load(os.path.join("voices", voice1_file), map_location=DEVICE)
            print(f"Successfully loaded first voice: {voice1_file}")
        except Exception as e:
            print(f"Error loading first voice {voice1_file}: {e}")
            return None
            
        try:
            voice2 = torch.load(os.path.join("voices", voice2_file), map_location=DEVICE)
            print(f"Successfully loaded second voice: {voice2_file}")
        except Exception as e:
            print(f"Error loading second voice {voice2_file}: {e}")
            return None
        
        # Linear interpolation
        print(f"Interpolating voices with ratio {ratio:.2f}")
        return voice1 * (1 - ratio) + voice2 * ratio
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None

def save_custom_voice(voice_tensor: torch.Tensor, voice_name: str) -> bool:
    """Save the custom voice tensor to a file."""
    try:
        # Ensure voice name has .pt extension
        if not voice_name.endswith('.pt'):
            voice_name = f"{voice_name}.pt"
            
        # Save to voices directory
        save_path = os.path.join("voices", voice_name)
        torch.save(voice_tensor, save_path)
        print(f"\nSuccessfully saved custom voice to: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving custom voice: {e}")
        return False

def main():
    try:
        print(f"Using device: {DEVICE}")
        voices = list_available_voices()
        
        while True:
            # Select first voice
            print("\nSelect the first voice:")
            voice1 = select_voice(voices, "\nSelect first voice number: ")
            
            # Select second voice
            print("\nSelect the second voice:")
            voice2 = select_voice(voices, "\nSelect second voice number: ")
            
            # Get interpolation ratio
            ratio = get_interpolation_ratio()
            
            # Get custom voice name
            voice_name = get_custom_voice_name()
            
            # Interpolate voices
            mixed_voice = interpolate_voices(voice1, voice2, ratio)
            if mixed_voice is None:
                print("Failed to interpolate voices")
                continue
            
            # Save the custom voice
            if save_custom_voice(mixed_voice, voice_name):
                print(f"\nYou can now use '{voice_name}' as a voice option in the TTS system!")
                print(f"\nTo try your new custom voice, run 'python audio_book.py' and select '{voice_name}' number from the voice number list!")
            
            # Ask if user wants to create another voice
            if input("\nCreate another custom voice? (y/n): ").lower().strip() != 'y':
                break
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()