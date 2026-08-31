"""Record computer audio and prepare WAV files for Whisper."""
import wave

from robot.config import MIC_DEVICE

SPEECH_RMS = 200   # a 100 ms block above this counts as speech
TRAIL_QUIET = 0.9  # seconds of quiet after speech before recording stops


def record_wav(path, max_seconds=8.0):
    """Record speech at the device rate and save a 16 kHz mono WAV.

    Recording stops after speech followed by quiet, or at `max_seconds`.
    Returns whether the input crossed the speech threshold.
    """
    import numpy as np
    import sounddevice as sd
    if MIC_DEVICE is not None:
        sd.default.device = (MIC_DEVICE, None)
    rate = int(sd.query_devices(kind="input")["default_samplerate"])
    block = int(rate * 0.1)
    chunks, spoke, quiet, lost = [], False, 0.0, False
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                        blocksize=block) as stream:
        for _ in range(int(max_seconds / 0.1)):
            data, overflowed = stream.read(block)
            lost = lost or overflowed
            chunks.append(data[:, 0].copy())
            rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
            if rms >= SPEECH_RMS:
                spoke, quiet = True, 0.0
            elif spoke:
                quiet += 0.1
                if quiet >= TRAIL_QUIET:
                    break
    if lost:
        print("mic buffer overflowed; some audio was dropped")
    samples = np.concatenate(chunks)
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
    return spoke


def wav_rms(path):
    """Loudness of a whole WAV, printed in the log after every recording."""
    import numpy as np
    with wave.open(str(path)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def is_silent(path, rms_floor=120):
    """Return whether a WAV is too quiet to transcribe reliably."""
    return wav_rms(path) < rms_floor


def trim_to_speech(path, pad=0.2):
    """Trim quiet ends in place and return the kept duration when changed."""
    import numpy as np
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            return None      # both writers make 16-bit mono; don't guess
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    block = max(1, rate // 10)          # 100 ms, as in record_wav
    blocks = len(samples) // block
    if not blocks:
        return None
    level = np.sqrt(np.mean(
        samples[:blocks * block].astype(np.float64).reshape(blocks, -1) ** 2,
        axis=1))
    loud = np.flatnonzero(level >= SPEECH_RMS)
    if not len(loud):
        return None
    keep = max(1, int(pad * rate) // block)
    a = max(0, loud[0] - keep) * block
    b = min(len(samples), (loud[-1] + 1 + keep) * block)
    if b - a >= len(samples):
        return None
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples[a:b].tobytes())
    return (b - a) / rate
