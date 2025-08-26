import sys, queue, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel
from collections import deque

MODEL_NAME = "tiny"   # tiny/base/small/medium/large-v3 (bigger = better but slower)
DEVICE = "cpu"         # "cuda" if you have an NVIDIA GPU
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.5    # capture block
WINDOW_SECONDS = 5.0   # rolling window to transcribe

def main():
    print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE}…")
    model = WhisperModel(MODEL_NAME, device=DEVICE)

    q = queue.Queue()
    ring = deque(maxlen=int(WINDOW_SECONDS / BLOCK_SECONDS))
    printed_upto = 0  # characters printed from last transcript

    def audio_cb(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # convert to float32 mono numpy at 16k
        audio = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        q.put(audio)

    print("🎤 Speak (Ctrl+C to stop)…")
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="int16",
                        blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
                        callback=audio_cb):
        try:
            while True:
                audio = q.get()
                ring.append(audio)
                # only transcribe when we have enough audio to be useful
                if len(ring) < ring.maxlen:
                    continue

                window_audio = np.concatenate(list(ring))
                segments, info = model.transcribe(
                    window_audio,
                    language=None,            # auto-detect
                    vad_filter=True,          # basic VAD to cut silence
                    beam_size=5,
                )

                text = "".join(s.text for s in segments).strip()
                if not text:
                    continue

                # print only the newly recognized tail to reduce flicker
                if printed_upto <= len(text):
                    new = text[printed_upto:]
                    if new:
                        print(new, end="", flush=True)
                    printed_upto = len(text)

        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()