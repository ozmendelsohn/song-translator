import os
import shutil
import subprocess
import torch
from pathlib import Path
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech
import torchaudio

# Global model cache
SEAMLESS_PROCESSOR = None
SEAMLESS_MODEL = None

def load_seamless_model():
    """Loads the SeamlessM4T v2 model if not already loaded."""
    global SEAMLESS_PROCESSOR, SEAMLESS_MODEL
    if SEAMLESS_MODEL is None:
        print("Loading SeamlessM4T v2 model...")
        model_name = "facebook/seamless-m4t-v2-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for SeamlessM4T: {device}")

        SEAMLESS_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        SEAMLESS_MODEL = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(model_name).to(device)

    return SEAMLESS_PROCESSOR, SEAMLESS_MODEL

def process_audio_seamless(audio_path, target_lang="spa", output_path="translated_vocals.wav"):
    """
    Translates audio to target language using SeamlessM4T v2.
    Returns path to translated audio.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio file not found or empty: {audio_path}")

    print(f"Translating audio {audio_path} to {target_lang}...")

    # Map language codes
    # Simplified mapping based on common usage
    lang_map = {
        "es": "spa",
        "fr": "fra",
        "de": "deu",
        "it": "ita",
        "pt": "por",
        "nl": "nld",
        "ru": "rus",
        "ja": "jpn",
        "ko": "kor",
        "zh": "cmn", # Mandarin Chinese
        "en": "eng"
    }

    if target_lang not in lang_map:
        raise ValueError(f"Unsupported target language: {target_lang}. Supported: {list(lang_map.keys())}")

    tgt_lang_code = lang_map[target_lang]

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

    processor, model = load_seamless_model()
    device = model.device

    # Resample to 16kHz required by SeamlessM4T
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # SeamlessM4T expects input shape (batch, time) or (time) depending on processor
    # Processor handles raw audio array. If waveform is (channels, time), we usually mix to mono.
    if waveform.shape[0] > 1:
        # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Squeeze to (time) if needed, processor expects array or tensor
    inputs = processor(audio=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

    print("Running S2ST inference...")
    # Generate translated speech
    with torch.no_grad():
        output = model.generate(**inputs, tgt_lang=tgt_lang_code)

    # Output[0] is the waveform tensor
    translated_waveform = output[0].cpu()

    # Save to file
    # torchaudio.save expects (channels, time)
    if translated_waveform.dim() == 1:
        translated_waveform = translated_waveform.unsqueeze(0)

    torchaudio.save(output_path, translated_waveform, 16000)

    return output_path

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

def detect_leading_silence(waveform, sample_rate, silence_threshold_db=-50.0, chunk_size_ms=10):
    """
    waveform: torch.Tensor (channels, time)
    silence_threshold_db: float
    chunk_size_ms: int
    Returns start_time in seconds (float)
    """
    # Convert threshold to amplitude
    # dB = 20 * log10(amplitude) -> amplitude = 10 ** (dB / 20)
    # Note: waveform is usually -1.0 to 1.0.
    threshold = 10 ** (silence_threshold_db / 20)

    chunk_samples = int(sample_rate * chunk_size_ms / 1000)
    if chunk_samples == 0:
        return 0.0

    # If stereo, take max amplitude across channels
    if waveform.shape[0] > 1:
        wave_mono = torch.max(torch.abs(waveform), dim=0)[0]
    else:
        wave_mono = torch.abs(waveform[0])

    length = wave_mono.shape[0]

    # Iterate in chunks to find start
    # Pad to multiple of chunk_samples
    pad_len = (chunk_samples - (length % chunk_samples)) % chunk_samples
    if pad_len > 0:
        wave_mono = torch.nn.functional.pad(wave_mono, (0, pad_len))

    num_chunks = wave_mono.shape[0] // chunk_samples
    chunks = wave_mono.view(num_chunks, chunk_samples)

    # Max amplitude per chunk
    max_per_chunk = torch.max(chunks, dim=1)[0]

    # Find first chunk > threshold
    indices = (max_per_chunk > threshold).nonzero(as_tuple=True)[0]

    if len(indices) == 0:
        return length / sample_rate # All silent

    first_chunk_idx = indices[0].item()
    start_time = first_chunk_idx * chunk_size_ms / 1000.0
    return start_time

def mix_audio(vocals_path, instrumental_path, start_time=0, output_path="final_mix.wav"):
    print(f"Mixing {vocals_path} and {instrumental_path} with offset {start_time}s...")

    vocals, sr_v = torchaudio.load(vocals_path)
    instrumental, sr_i = torchaudio.load(instrumental_path)

    # Resample vocals if needed
    if sr_v != sr_i:
        resampler = torchaudio.transforms.Resample(sr_v, sr_i)
        vocals = resampler(vocals)
        sr_v = sr_i

    # Vocals volume boost with clipping protection
    peak = torch.max(torch.abs(vocals))
    if peak > 0:
        gain = min(1.5, 1.0 / peak)
    else:
        gain = 1.0
    vocals = vocals * gain

    # Offset
    offset_samples = int(start_time * sr_i)

    # Create empty tensor for mix
    # Length is max(instrumental length, offset + vocals length)
    max_len = max(instrumental.shape[1], offset_samples + vocals.shape[1])

    mix = torch.zeros((instrumental.shape[0], max_len))

    # Add instrumental
    mix[:, :instrumental.shape[1]] += instrumental

    # Add vocals
    # Handle channel mismatch
    target_channels = instrumental.shape[0]

    # If vocals mono, expand to stereo if instrumental is stereo
    if vocals.shape[0] == 1 and target_channels == 2:
        vocals = vocals.repeat(2, 1)
    elif vocals.shape[0] > target_channels:
        # Explicit downmix by averaging channels
        vocals = torch.mean(vocals, dim=0, keepdim=True)
        # If target is more than 1 (e.g. instrumental is 2, vocals was >2 and now 1), expand again
        if target_channels > 1:
            vocals = vocals.repeat(target_channels, 1)

    # Ensure vocals fits in mix bounds
    end_sample = offset_samples + vocals.shape[1]

    # Add to mix
    # Channels should now match or be broadcastable, but we force slice to be safe
    # vocals should match target_channels due to logic above
    if vocals.shape[0] == target_channels:
        mix[:, offset_samples:end_sample] += vocals
    else:
        # Should ideally not happen with above logic, but as failsafe:
        print("Warning: Channel mismatch persisted in mixing, slicing")
        mix[:min(vocals.shape[0], target_channels), offset_samples:end_sample] += vocals[:min(vocals.shape[0], target_channels)]

    # Clip to -1.0, 1.0
    mix = torch.clamp(mix, -1.0, 1.0)

    # Save as WAV
    torchaudio.save(output_path, mix, sr_i)
    return output_path

def process_song(audio_file, target_lang):
    try:
        if audio_file is None:
            return None, None, None, "No file uploaded"

        # 1. Separate
        vocals_path, no_vocals_path = separate_vocals(audio_file)

        # 2. Trim silence from vocals to optimize S2ST
        vocals, sr = torchaudio.load(vocals_path)
        start_time_sec = detect_leading_silence(vocals, sr)

        # If the whole track is silent?
        duration = vocals.shape[1] / sr
        if start_time_sec >= duration:
            print("Vocals seem silent.")
            return no_vocals_path, vocals_path, "No vocals detected", "No translation"

        # Create trimmed version
        start_sample = int(start_time_sec * sr)
        trimmed_vocals = vocals[:, start_sample:]

        vocals_path_obj = Path(vocals_path)
        trimmed_vocals_path = str(vocals_path_obj.parent / "vocals_trimmed.wav")
        torchaudio.save(trimmed_vocals_path, trimmed_vocals, sr)

        print(f"Vocals start at {start_time_sec:.2f} seconds")

        # 3. Translate Speech-to-Speech
        translated_vocals_path = str(vocals_path_obj.parent / "translated_vocals.wav")
        process_audio_seamless(trimmed_vocals_path, target_lang, output_path=translated_vocals_path)

        # 4. Mix
        final_mix_path = mix_audio(translated_vocals_path, no_vocals_path, start_time=start_time_sec)

        return (
            final_mix_path,
            vocals_path,
            "Lyrics extraction not supported in this mode",
            f"Translation is Speech-to-Speech via SeamlessM4T ({target_lang})"
        )
    except Exception as e:
        print(f"Error processing song: {e}")
        import traceback
        traceback.print_exc()
        return None, None, str(e), "Error during processing"
