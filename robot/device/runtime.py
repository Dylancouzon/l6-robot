"""Small operating-system-neutral helpers for the live device app."""
import os
import tempfile
import time
from pathlib import Path

# Use the platform's temporary directory instead of assuming a Unix /tmp.
# The process id also keeps two simultaneously running robots from sharing a
# recording, which matters on multi-user desktop machines.
UTTERANCE_WAV = str(
    Path(tempfile.gettempdir()) / f"qdrant-memory-robot-{os.getpid()}.wav")


def _clock(t):
    """12-hour clock without a leading zero, on Unix and Windows."""
    return time.strftime("%I:%M %p", t).lstrip("0")


def when(ts, now=None):
    """A spoken timestamp, with no platform-specific strftime directives.

    Today keeps just the clock; older sightings name the day, because "I saw
    it at 9:12 PM" is a lie by omission on Tuesday.
    """
    t = time.localtime(ts)
    today = time.localtime(time.time() if now is None else now)
    clock = _clock(t)
    if (t.tm_year, t.tm_yday) == (today.tm_year, today.tm_yday):
        return clock
    return f"{time.strftime('%b', t)} {t.tm_mday}, {clock}"


def stamp(ts):
    """Compact UI timestamp, portable across C runtime implementations."""
    t = time.localtime(ts)
    return f"{time.strftime('%b', t)} {t.tm_mday}, {time.strftime('%H:%M', t)}"
