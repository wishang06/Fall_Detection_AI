from faster_whisper import WhisperModel

# model sizes: tiny, base, small, medium, large-v3 (bigger = better but slower)
model = WhisperModel("small", device="cpu")   # use device="cuda" if you have NVIDIA GPU

segments, info = model.transcribe("example.wav", beam_size=5)
print(f"Detected language: {info.language} (prob {info.language_probability:.2f})")
text = "".join(seg.text for seg in segments)
print(text)