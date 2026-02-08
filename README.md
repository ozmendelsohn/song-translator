# AI Song Translator PoC

This project is a Proof-of-Concept (PoC) for translating songs from one language to another while preserving the original instrumental track. It leverages state-of-the-art open-source AI models for source separation, transcription, translation, and speech synthesis.

## Features

*   **Source Separation:** Isolates vocals from the instrumental track using **Demucs** (Hybrid Transformer).
*   **Transcription (ASR):** Transcribes the original vocals to text and captures timing using **OpenAI Whisper**.
*   **Translation:** Translates lyrics to the target language using **NLLB-200** (No Language Left Behind) via Hugging Face Transformers.
*   **Synthesis (TTS):** Generates spoken/sung vocals in the target language using **Edge-TTS** (Microsoft Edge Online TTS).
*   **Mixing:** Recombines the new vocals with the original instrumental track, aligned to the start of the original singing.
*   **Web UI:** Simple and interactive interface built with **Gradio**.

## Prerequisites

*   **Python 3.10+** (Managed automatically if using `uv`)
*   **FFmpeg:** Required for audio processing.
    *   **Linux (Ubuntu/Debian):** `sudo apt install ffmpeg`
    *   **macOS:** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
*   **uv:** Fast Python package installer and resolver.
    *   Install via pip: `pip install uv`
    *   Or via curl: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```
    This will create a virtual environment at `.venv` and install all required packages defined in `pyproject.toml`.

## Usage

1.  **Run the application:**
    ```bash
    uv run app.py
    ```
    Or manually activate the environment:
    ```bash
    source .venv/bin/activate
    python app.py
    ```

2.  **Open the Web UI:**
    The application will launch a local server, typically at `http://127.0.0.1:7860`. Open this URL in your browser.

3.  **Translate a Song:**
    *   Upload an audio file (MP3, WAV, etc.).
    *   Select the target language from the dropdown.
    *   Click **"Translate & Sing"**.
    *   Wait for the process to complete (this may take a minute or two depending on your hardware).
    *   Download the "Final Translated Mix" or listen to the isolated tracks.

## Project Structure

*   `app.py`: The Gradio frontend application.
*   `backend.py`: Core logic pipeline (Separation -> Transcription -> Translation -> TTS -> Mixing).
*   `REPORT.md`: Detailed research report on the methodology and findings.
*   `pyproject.toml` / `uv.lock`: Dependency definitions.

## Models Used

*   **Demucs:** `htdemucs` (Hybrid Transformer)
*   **Whisper:** `base` model (can be upgraded in `backend.py`)
*   **NLLB:** `facebook/nllb-200-distilled-600M`
*   **Edge-TTS:** `en-US-ChristopherNeural`, `es-ES-AlvaroNeural`, etc.

## Limitations

*   **Rhythm & Melody:** This PoC uses a "dubbing" approach. The translated vocals are spoken/chanted with the rhythm of the TTS engine, not the original melody of the song. Preserving the exact melody with translated lyrics is a complex research problem (Singing Voice Synthesis).
*   **Performance:** Source separation and NLLB translation can be resource-intensive. A GPU (NVIDIA CUDA) is highly recommended for faster processing.
*   **Translation Accuracy:** Song lyrics are poetic and difficult to translate literally. NLLB provides a good baseline but may miss nuances.

## Troubleshooting

*   **FFmpeg Error:** If you see errors related to audio loading/saving, ensure `ffmpeg` is installed and accessible in your system PATH.
*   **CUDA/GPU:** If you have a GPU but it's not being used, ensure you have the correct NVIDIA drivers installed. `uv` installs the CPU version of PyTorch by default on some platforms; you may need to override this in `pyproject.toml` or install the CUDA version manually if needed.
