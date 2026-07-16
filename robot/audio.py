"""Push-to-talk mic capture (PortAudio) and label parsing.

ffmpeg/avfoundation was tried first and failed to open two different USB
audio devices ("Cannot use ..."); sounddevice handles them fine.
"""
import re
import wave

MIC_DEVICE = None  # None = system default input. To pick another device:
                   # uv run python -c "import sounddevice; print(sounddevice.query_devices())"
                   # and set this to the device index or name.


def record_wav(path, seconds=5.0):
    """Record the mic to a 16 kHz mono WAV — what Whisper expects.

    Records at the device's native rate (USB mics often refuse 16 kHz),
    then resamples; linear interpolation is plenty for speech.
    """
    import numpy as np
    import sounddevice as sd
    if MIC_DEVICE is not None:
        sd.default.device = (MIC_DEVICE, None)
    rate = int(sd.query_devices(kind="input")["default_samplerate"])
    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                  dtype="int16")
    sd.wait()
    samples = data[:, 0]
    if rate != 16000:
        n = int(len(samples) * 16000 / rate)
        samples = np.interp(
            np.linspace(0, len(samples), n, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())
    return str(path)


def wav_rms(path):
    import numpy as np
    with wave.open(str(path)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def is_silent(path, rms_floor=120):
    """True if the WAV is near-silence. Whisper hallucinates on silence
    ("Thanks for watching!"), so a failed mic capture must fail visibly
    instead of teaching a nonsense label. All-zero audio also means macOS
    hasn't granted the terminal microphone permission."""
    return wav_rms(path) < rms_floor


# ASR often drops the punctuation that would end the naming clause, so the
# label also stops at words that start a new clause ("...my mug I bought it").
_CLAUSE_WORDS = {"i", "it", "and", "that", "which", "because", "she", "he",
                 "they", "we", "you", "made", "bought", "got", "from", "at",
                 "about", "in", "on"}


def parse_label(transcript):
    """'This is my mug — Maria made it.' → 'my mug'. Free-form fallback."""
    m = re.search(r"this is (?:an? )?(.+?)(?:\s*[,.;!?—–-]|$)",
                  transcript, re.IGNORECASE)
    phrase = m.group(1) if m else transcript
    words = []
    for w in phrase.split():
        if w.lower().strip(".,!?") in _CLAUSE_WORDS:
            break
        words.append(w)
    return " ".join(words[:5]).rstrip(".,!?") or "unnamed"
