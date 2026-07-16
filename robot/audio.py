"""Push-to-talk mic capture (ffmpeg/avfoundation) and label parsing."""
import re
import subprocess

MIC_DEVICE = ":0"  # default mic; list devices with:
                   # ffmpeg -f avfoundation -list_devices true -i ""


def record_wav(path, seconds=5.0):
    """Record the mic to a 16 kHz mono WAV — what Whisper expects."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "avfoundation", "-i", MIC_DEVICE,
         "-t", str(seconds), "-ar", "16000", "-ac", "1", "-y", str(path)],
        check=True,
    )
    return str(path)


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
