# Song Translator PoC

A proof-of-concept application that translates song lyrics and synthesizes them back into the song.

## Features

- **Separate Vocals**: Uses Demucs to separate vocals from the instrumental.
- **Transcribe**: Uses Whisper to transcribe the vocals.
- **Translate**: Uses NLLB to translate the lyrics to the target language.
- **Synthesize**: Uses Edge-TTS to synthesize the translated lyrics.
- **Mix**: Mixes the new vocals with the original instrumental.

## Usage

1. Run the application: `python app.py`
2. Open the Gradio interface in your browser.
3. Upload an audio file.
4. Select the target language.
5. Click "Translate & Sing".
