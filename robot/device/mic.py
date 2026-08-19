"""Microphone capture on the machine running the app, for the laptop path.

The Jetson has no audio hardware: on the appliance the phone browser records
and uploads a WAV instead, and only the helpers below that read a file are
used. ffmpeg/avfoundation failed to open two different USB audio devices, so
capture goes through sounddevice.
"""
import wave

MIC_DEVICE = None  # None = system default input. To pick another device:
                   # uv run python -c "import sounddevice; print(sounddevice.query_devices())"
                   # and set this to the device index or name.

SPEECH_RMS = 200   # a 100 ms block above this counts as speech
TRAIL_QUIET = 0.9  # seconds of quiet after speech before recording stops


def record_wav(path, max_seconds=8.0):
    """Record the mic to a 16 kHz mono WAV, which is what Whisper expects.

    Stops on its own once speech is followed by ~a second of quiet;
    `max_seconds` is the cap, not the duration. Records at the device's
    native rate (USB mics often refuse 16 kHz) and resamples.

    Returns True if speech crossed SPEECH_RMS, so the caller does not have to
    re-derive silence from the clip.
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
        # a garbled transcript after this line is dropped audio, not Whisper:
        # something starved this loop while the stream was open
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
    """True if the WAV is near-silence.

    Whisper hallucinates on silence ("Thanks for watching!"), so a failed
    capture has to fail visibly instead of teaching a nonsense label. All-zero
    audio also means the mic permission was never granted. Used for the phone
    path, whose uploaded WAV carries no speech flag; record_wav returns its own.
    """
    return wav_rms(path) < rms_floor


def trim_to_speech(path, pad=0.2):
    """Cut the quiet off both ends in place. Returns the kept duration, or None.

    Whisper charges for every second it is given and starts repeating itself
    when handed silence, so the recording that reaches it should be the
    utterance rather than the whole button hold.

    Fails open, and uses the same SPEECH_RMS bar record_wav stops on, so both
    mic paths agree on what counts as speech. A hold too quiet to reach that
    bar is passed through whole: guessing where the speech was in a clip that
    quiet is worse than the rms line in the log telling the operator to speak up.
    """
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
