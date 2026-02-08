import os
import shutil
import asyncio
import subprocess
import torch
import whisper
from deep_translator import GoogleTranslator
import edge_tts
from pydub import AudioSegment

# Global model cache
WHISPER_MODEL = None

def setup_pipeline():
    """Loads the Whisper model if not already loaded."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        WHISPER_MODEL = whisper.load_model("base", device=device)
    return WHISPER_MODEL

def separate_vocals(audio_path, output_dir="separated"):
    """
    Separates vocals from the audio using Demucs.
    Returns paths to (vocals, no_vocals).
    """
    print(f"Separating vocals for: {audio_path}")

    model_name = "htdemucs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run Demucs via subprocess
    # Use --two-stems=vocals to get vocals.wav and no_vocals.wav directly
    cmd = [
        "demucs",
        "-o", output_dir,
        "-n", model_name,
        "--device", device,
        "--two-stems=vocals",
        audio_path
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Determine output paths
    # Demucs output structure: output_dir/model_name/filename_no_ext/vocals.wav
    filename_with_ext = os.path.basename(audio_path)
    filename_no_ext = os.path.splitext(filename_with_ext)[0]

    # Demucs might sanitize filename, but usually it's just the name
    result_dir = os.path.join(output_dir, model_name, filename_no_ext)

    vocals_path = os.path.join(result_dir, "vocals.wav")
    no_vocals_path = os.path.join(result_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        # Fallback: check if directory exists, maybe filename mismatch
        potential_dirs = [d for d in os.listdir(os.path.join(output_dir, model_name)) if filename_no_ext in d]
        if potential_dirs:
            result_dir = os.path.join(output_dir, model_name, potential_dirs[0])
            vocals_path = os.path.join(result_dir, "vocals.wav")
            no_vocals_path = os.path.join(result_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocals not found at {vocals_path}")

    return vocals_path, no_vocals_path

def transcribe(audio_path):
    """
    Transcribes audio using Whisper.
    Returns the text and the start time of the first segment (in seconds).
    """
    model = setup_pipeline()
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)
    text = result["text"].strip()

    start_time = 0.0
    if result.get("segments"):
        # Use the start time of the first segment
        start_time = result["segments"][0]["start"]

    return text, start_time

def translate(text, target_lang="es"):
    """
    Translates text to target language.
    """
    print(f"Translating to {target_lang}...")
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation failed: {e}")
        return text # Fallback to original text

async def run_tts_async(text, voice, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
    except Exception as e:
        print(f"Error inside run_tts_async: {e}")
        # Re-raise so calling code knows it failed
        raise e

def synthesize(text, target_lang="es"):
    """
    Synthesizes speech from text using Edge TTS.
    Returns path to generated audio.
    """
    print(f"Synthesizing speech for: {text[:20]}...")
    output_file = "tts_output.mp3"

    # Map language to a voice
    voice_map = {
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-KillianNeural",
        "it": "it-IT-DiegoNeural",
        "pt": "pt-BR-AntonioNeural",
        "nl": "nl-NL-MaartenNeural",
        "ru": "ru-RU-DmitryNeural",
        "ja": "ja-JP-KeitaNeural",
        "ko": "ko-KR-InJoonNeural",
        "zh": "zh-CN-YunxiNeural",
        "en": "en-US-ChristopherNeural"
    }

    voice = voice_map.get(target_lang, "en-US-ChristopherNeural")

    # Run async function in sync wrapper
    try:
        # asyncio.run() creates a new loop and closes it. It requires no running loop in current thread.
        # Gradio functions run in a thread pool, so no loop should be active.
        asyncio.run(run_tts_async(text, voice, output_file))
    except Exception as e:
        print(f"Error in synthesize: {e}")
        raise e

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("TTS output file is empty or not found")

    return output_file

def mix_audio(vocals_path, instrumental_path, start_time=0, output_path="final_mix.mp3"):
    """
    Mixes vocals and instrumental.
    start_time: offset in seconds for the vocals (TTS).
    """
    print(f"Mixing {vocals_path} and {instrumental_path} with offset {start_time}s...")

    # Load audio segments
    vocals = AudioSegment.from_file(vocals_path) # Might be mp3
    instrumental = AudioSegment.from_file(instrumental_path) # Might be wav

    # Adjust volume of vocals slightly up?
    vocals = vocals + 2

    # Calculate position in milliseconds
    position = int(start_time * 1000)

    # Overlay vocals on instrumental
    mixed = instrumental.overlay(vocals, position=position)

    mixed.export(output_path, format="mp3")
    return output_path

def process_song(audio_file, target_lang):
    try:
        if audio_file is None:
            return None, None, None, "No file uploaded"

        # 1. Separate
        vocals_path, no_vocals_path = separate_vocals(audio_file)

        # 2. Transcribe
        original_lyrics, start_time = transcribe(vocals_path)

        if not original_lyrics:
            print("No lyrics detected.")
            original_lyrics = "[No lyrics detected]"
            translated_lyrics = "[No translation]"
            # Return instrumental only as final mix
            return no_vocals_path, vocals_path, original_lyrics, translated_lyrics

        # 3. Translate
        translated_lyrics = translate(original_lyrics, target_lang)

        # 4. Synthesize
        tts_audio_path = synthesize(translated_lyrics, target_lang)

        # 5. Mix
        # Use the start_time from Whisper to align the TTS audio
        final_mix_path = mix_audio(tts_audio_path, no_vocals_path, start_time=start_time)

        return (
            final_mix_path,
            vocals_path,
            original_lyrics,
            translated_lyrics
        )
    except Exception as e:
        print(f"Error processing song: {e}")
        import traceback
        traceback.print_exc()
        return None, None, str(e), "Error during processing"
