# Report: Automated Song Translation with Off-the-Shelf Models

## Executive Summary
This report outlines a strategy for translating songs from one language to another while attempting to preserve the original musical elements (melody, structure, beat, timbre). The proposed solution leverages state-of-the-art, open-source AI models available on Hugging Face and other repositories, focusing on ease of implementation and local execution capabilities.

## Problem Analysis
Translating a song involves several distinct challenges:
1.  **Source Separation:** Isolating the vocal track from the instrumental accompaniment.
2.  **Transcription (ASR):** Accurately converting the singing voice to text (lyrics) and capturing timing information.
3.  **Translation:** Converting lyrics to the target language. A key challenge here is **rhythm and meter preservation**—translated lyrics often have different syllable counts and stress patterns than the original, making them difficult to sing to the original melody without adaptation.
4.  **Synthesis (TTS/SVS):** Generating singing voice in the target language.
5.  **Voice Conversion (VC):** Modifying the synthesized voice to match the timbre of the original singer.
6.  **Mixing:** Recombining the new vocals with the original instrumental track.

## Recommended Pipeline (Proof of Concept)

We propose a modular pipeline using the following "off-the-shelf" components:

### 1. Source Separation: `Demucs` (Hybrid Transformer Demucs)
- **Role:** Isolate vocals from the instrumental backing track.
- **Why:** State-of-the-art quality, widely used, open-source (Facebook Research).
- **Alternative:** `Spleeter` (faster but lower quality).

### 2. Transcription: `Whisper` (OpenAI)
- **Role:** Transcribe the isolated vocal track to get the original lyrics and timestamps.
- **Why:** Robust performance on diverse audio, including singing; provides word-level timestamps which are crucial for alignment.

### 3. Translation: `NLLB` (No Language Left Behind) or `Google Translate` API
- **Role:** Translate lyrics to the target language.
- **Why:** `NLLB` is a high-quality open model from Meta. For simplicity in a PoC, `googletrans` (unofficial API) or a simple HF model can be used.
- **Note:** Standard translation does not preserve rhyme or meter. This is a complex research problem. For a PoC, we will accept a "literal" translation or use a Large Language Model (LLM) to attempt rhythmic matching.

### 4. Synthesis & Voice Conversion: `Edge-TTS` + `RVC` (Retrieval-based Voice Conversion)
- **Role:** Generate the singing/spoken voice in the target language.
- **Why:**
    - `Edge-TTS` is a free, high-quality TTS engine that supports many languages. It generates speech, not singing.
    - `RVC` is the current standard for AI voice cloning. It can take an audio input (the TTS speech) and convert its timbre to match a target voice (the original singer).
    - **Challenge:** To truly "sing", we need a Singing Voice Synthesis (SVS) model. However, high-quality generic SVS models that accept arbitrary text and melody are rare and complex to set up.
    - **PoC Compromise:** The initial PoC will likely result in a "spoken word" or "melodic speech" version of the translation, as true singing synthesis requires aligning phonemes to musical notes, which is non-trivial without manual intervention.

### 5. Mixing: `Pydub` / `FFmpeg`
- **Role:** Combine the new vocal track with the original instrumental.

## Implementation Strategy

### Frontend: Gradio
We will build a web-based UI using **Gradio**. It is:
- **Easy to implement:** Python-based, requires minimal frontend code.
- **Interactive:** Allows users to upload audio, select target languages, and play back results.
- **Local:** Runs locally in the browser or can be hosted.

### Backend: Python
The backend will orchestrate the pipeline:
1.  Receive audio file.
2.  Run `demucs` to split stems.
3.  Run `whisper` on `vocals.wav` to get text.
4.  Translate text.
5.  Run TTS on translated text to generate `tts_vocals.wav`.
    - *Advanced:* Stretch/warp `tts_vocals.wav` to match the duration of original phrases.
6.  (Optional) Run RVC to style transfer `tts_vocals.wav` to look like `vocals.wav`.
7.  Mix `tts_vocals.wav` with `no_vocals.wav`.
8.  Return the final audio.

## Technical Requirements (Local Execution)
- **Python 3.8+**
- **FFmpeg** (installed on system)
- **GPU (CUDA):** Highly recommended for `Demucs`, `Whisper`, and `RVC`. Can run on CPU but will be slow.
- **Disk Space:** ~5-10GB for models.

## Future Improvements / WebAssembly
While the user requested "WebAssembly using GPU", running heavy models like `Demucs` (separation) and `RVC` (voice conversion) entirely client-side via WASM is currently experimental and performance-heavy.
- **Partial WASM:** `Whisper` and some TTS models can run in-browser via `Transformers.js`.
- **Full WASM:** A full pipeline would likely crash a browser tab due to memory/compute limits.
- **Recommendation:** A local Python server with a lightweight web UI is the most robust and "easiest to implement" solution today.
