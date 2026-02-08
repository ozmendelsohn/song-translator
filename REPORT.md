# Report: Automated Song Translation with Hugging Face Models

## Executive Summary
This report outlines a strategy for translating songs from one language to another using state-of-the-art open-source AI models, primarily leveraging the Hugging Face ecosystem. The solution focuses on preserving the original instrumental track while replacing the vocals with a translated version.

## Problem Analysis
Translating a song involves several distinct challenges:
1.  **Source Separation:** Isolating the vocal track from the instrumental accompaniment.
2.  **Transcription (ASR):** Accurately converting the singing voice to text (lyrics) and capturing timing information.
3.  **Translation:** Converting lyrics to the target language. A key challenge here is **rhythm and meter preservation**.
4.  **Synthesis (TTS):** Generating singing/spoken voice in the target language.
5.  **Mixing:** Recombining the new vocals with the original instrumental track.

## Recommended Pipeline (Proof of Concept)

We propose a modular pipeline using the following components:

### 1. Source Separation: `Demucs` (Hybrid Transformer Demucs)
- **Role:** Isolate vocals from the instrumental backing track.
- **Why:** State-of-the-art quality, widely used, available via `torchaudio` or standalone CLI.

### 2. Transcription: `Whisper` (OpenAI via Hugging Face or PyPI)
- **Role:** Transcribe the isolated vocal track to get the original lyrics and timestamps.
- **Why:** Robust performance on diverse audio; provides word-level timestamps crucial for alignment.

### 3. Translation: `NLLB` (No Language Left Behind)
- **Model:** `facebook/nllb-200-distilled-600M`
- **Role:** Translate lyrics to the target language.
- **Why:** A high-quality, open-source multilingual translation model available on **Hugging Face**. It supports over 200 languages and runs locally, satisfying the requirement for "ML models from HF".

### 4. Synthesis: `Edge-TTS` (Interim Solution)
- **Role:** Generate the singing/spoken voice in the target language.
- **Why:** While Hugging Face offers models like **Bark** (`suno/bark`) or **MMS** (`facebook/mms-tts`), they are often computationally heavy or require complex setup for multilingual support. `Edge-TTS` provides high-quality, natural-sounding multilingual speech with minimal overhead, making it ideal for a "easiest to implement" PoC.
- **HF Alternative:** For a pure ML approach, `SpeechT5` (`microsoft/speecht5_tts`) or `Bark` could be swapped in, but with significant performance trade-offs.

### 5. Mixing: `Pydub` / `FFmpeg`
- **Role:** Combine the new vocal track with the original instrumental.
- **Alignment:** We use the start time detected by Whisper to offset the TTS audio, ensuring it starts when the original vocals did.

## Alternative: End-to-End Speech Translation
**SeamlessM4T (Meta):**
- **Model:** `facebook/seamless-m4t-v2-large`
- **Capability:** Direct Speech-to-Speech Translation (S2ST).
- **Pros:** Single model, handles translation and synthesis.
- **Cons:** It generates speech from scratch and does not separate or preserve the background music. Using it on a song would result in an a cappella translation or a hallucinated background.
- **Verdict:** Not suitable for *song* translation where preserving the original backing track is required.

## Implementation Strategy
- **Frontend:** Gradio (Web UI).
- **Backend:** Python script integrating `demucs`, `whisper`, `transformers` (NLLB), and `edge-tts`.
- **Dependency Management:** `uv` is used for fast and reliable package management.
- **Hardware:** GPU recommended for Demucs, Whisper, and NLLB.

## Future Improvements
- **Singing Voice Synthesis (SVS):** Replace `Edge-TTS` with a model trained specifically for singing (e.g., `Diff-SVC` or `So-VITS-SVC`), conditioned on the melody extracted from the original vocals.
- **Rhythm Adaptation:** Use an LLM to rewrite translated lyrics to match the syllable count of the original lines.
