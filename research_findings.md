# Research Findings: Song Translation Pipeline

## Objective
Translate a song from one language to another while preserving:
- Tones (Melody)
- Structure
- Beats
- Tempo
- Sound (Voice timbre)

## Challenges
1. **Rhythm & Syllable Mismatch:** Translated lyrics rarely have the same number of syllables or stress patterns as the original. Direct translation often breaks the flow.
2. **Melody Alignment:** Mapping the new syllables to the original melody notes requires complex alignment or a model that can generate singing from text + pitch curve.
3. **Voice Cloning:** Preserving the original singer's timbre requires Voice Conversion (VC) or Zero-Shot TTS.

## Potential Pipeline Components

### 1. Source Separation (Vocal Extraction)
- **Demucs (Hybrid Transformer Demucs):** State-of-the-art music source separation.
  - *Repo:* `facebookresearch/demucs`
  - *Pros:* High quality.
  - *Cons:* Computationally expensive (GPU recommended).
- **Spleeter:** Older, faster, but lower quality than Demucs.
- **Open-Unmix:** Another option.

### 2. Transcription (ASR)
- **Whisper (OpenAI):** Best in class for multilingual transcription.
  - *Repo:* `openai/whisper`
  - *Pros:* Handles music/singing reasonably well, timestamps are accurate.
  - *Cons:* Can hallucinate on instrumental breaks.

### 3. Translation
- **NLLB / M2M100 (Meta):** High-quality open-source translation models.
- **LLMs (Llama 2, GPT-4):** Can be prompted to "match rhythm" or "rhyme", which is crucial for songs.
  - *Note:* For "off-the-shelf" local execution, a small LLM or NLLB is best.

### 4. Singing Synthesis / Voice Conversion
This is the hardest part.
- **Option A: TTS + RVC (Retrieval-based Voice Conversion)**
  1. Generate speech/singing of translated lyrics using a TTS engine (e.g., Edge-TTS, Coqui TTS).
  2. Use RVC to convert the TTS voice to the original singer's voice.
  3. *Problem:* The TTS output won't follow the original melody/tempo.
- **Option B: Phoneme replacement + Pitch preservation**
  - Extract pitch (F0) from original vocals.
  - Extract phonemes from translated text.
  - Synthesize using a model that accepts Phonemes + F0 (like Diff-SVC or So-VITS-SVC).
  - *Problem:* Alignment. We need to know *when* each translated phoneme should be sung.
- **Option C: End-to-End Speech-to-Speech Translation**
  - **SeamlessM4T (Meta):** Can do S2ST (Speech-to-Speech Translation).
  - *Problem:* Trained on speech, might flatten melody.

## Recommended "Easiest to Implement" Approach for PoC
**The "Dubbing" Approach (Speech-focused translation, preserving background music):**
1. **Separate** vocals and drums/bass/other using `demucs`.
2. **Transcribe** vocals using `whisper`.
3. **Translate** text using `googletrans` or `sentence-transformers`.
4. **Synthesize** translated text using `edge-tts` (fast, decent quality).
5. **(Optional) Voice Convert** using a pre-trained RVC model or similar (might be too complex for a quick PoC).
6. **Mix** the new vocal track with the background tracks.

**Refinement for "Song" Translation:**
To actually "sing", we need a Singing TTS.
- **Bark (Suno):** Can generate singing, but control is hard.
- **ACE-Studio / Synthesizer V:** Commercial, high quality.
- **Open Source SVS:** `Opencpop`, `Diff-SVC`. Hard to set up.

**Conclusion for PoC:**
We will build a modular pipeline.
- **Frontend:** Gradio (Web UI).
- **Backend:**
  - `demucs` for separation.
  - `whisper` for ASR.
  - `googletrans` (or simple HF model) for translation.
  - `edge-tts` for generating the audio (spoken/dubbed style first, as singing is hard).
  - `pydub` for audio mixing.

This will demonstrate the full cycle. The "singing" part will likely sound like spoken word over music unless we use a specific Singing TTS, which is a project in itself. We can add a "Voice Conversion" step if time permits.
