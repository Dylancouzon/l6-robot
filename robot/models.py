"""The course's exact model stack: Nomic 768 (text), CLIP 512 (image), Whisper.

Same models and the same FastEmbed and onnx-asr loaders as the course
notebooks. The "same stack" claim in L6 depends on this file staying aligned
with the course's helper.py, and on the pinned versions in pyproject.toml:
the recognition threshold is calibrated against those exact encoders (and
against the crop pipeline — see testdata/verify_scores.py).

Each loader is cached, so a model is built once and only when something first
asks for it. That laziness is what lets an 8 GB Jetson run this: see warm_up.
"""
import os
import threading
from functools import lru_cache
from pathlib import Path

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_DIM = 768
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
CLIP_DIM = 512
WHISPER_MODEL = "whisper-base"

# ONNX Runtime gives every session one thread per core and lets idle threads
# spin. Four encoders next to the detector would then oversubscribe a Jetson's
# six cores and starve the camera loop — the feed freezes while Whisper thinks.
# Two threads each leaves the loop room to keep drawing; it costs Whisper about
# half a second per utterance.
ENCODER_THREADS = 2

# FastEmbed caches its ONNX files in /tmp by default, and /tmp is swept: on the
# Jetson, systemd-tmpfiles prunes it at 30 days. That is 1.1 GB of encoders the
# headless robot cannot fetch again, because on its own hotspot it has no
# internet — it would boot into a download that never finishes. Keep the cache
# somewhere durable instead. FASTEMBED_CACHE_PATH still wins if it is set;
# that is FastEmbed's own override, and it names the directory outright.
CACHE_DIR = (os.environ.get("FASTEMBED_CACHE_PATH")
             or str(Path.home() / ".cache" / "fastembed"))


def _sess_options():
    """ONNX Runtime's documented threading controls, for onnx-asr."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ENCODER_THREADS
    opts.inter_op_num_threads = ENCODER_THREADS
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


@lru_cache(maxsize=1)
def _text_model():
    from fastembed import TextEmbedding
    return TextEmbedding(NOMIC_MODEL, threads=ENCODER_THREADS,
                         cache_dir=CACHE_DIR)


@lru_cache(maxsize=1)
def _clip_vision():
    from fastembed import ImageEmbedding
    return ImageEmbedding(CLIP_VISION_MODEL, threads=ENCODER_THREADS,
                          cache_dir=CACHE_DIR)


@lru_cache(maxsize=1)
def _clip_text():
    from fastembed import TextEmbedding
    return TextEmbedding(CLIP_TEXT_MODEL, threads=ENCODER_THREADS,
                         cache_dir=CACHE_DIR)


@lru_cache(maxsize=1)
def _asr_model():
    import onnx_asr
    return onnx_asr.load_model(WHISPER_MODEL, sess_options=_sess_options(),
                               providers=["CPUExecutionProvider"])


def embed_text(text):
    """Embed one document (a transcript) for storage. Nomic, 768-d."""
    return next(_text_model().embed([text])).tolist()


def embed_query(text):
    """Embed one question. Nomic's query prefix matters for retrieval."""
    return next(_text_model().query_embed([text])).tolist()


def embed_crop(bgr):
    """Embed one OpenCV BGR crop with CLIP's vision tower. 512-d."""
    from PIL import Image
    img = Image.fromarray(bgr[:, :, ::-1])
    return next(_clip_vision().embed([img])).tolist()


def embed_query_clip(text):
    """Embed a question into CLIP's space, to search the image vector."""
    return next(_clip_text().query_embed([text])).tolist()


def transcribe(wav_path):
    """Local Whisper speech-to-text on one WAV file."""
    return _asr_model().recognize(wav_path).strip()


# warm_encoders is called from two threads per interaction — once at phone
# pointerdown (a background warm, so the loads run during the button hold) and
# once in _process before transcribing. lru_cache does not serialize a cache
# miss, so without this lock both threads would build the same model at once:
# double the load time, and a transient extra model's worth of memory on a
# board that has none to spare. embed_crop never contends: _clip_vision is
# built by warm_up() at startup and no path builds it here. The OTHER embed
# helpers (embed_text, embed_query, embed_query_clip) skip this lock and are
# safe only by call ordering — every caller runs warm_encoders(kind) first on
# its own thread, so their models are always cached by the time they run. Embed
# text from any new thread without that warm call and the double build is back.
_warm_lock = threading.Lock()


def warm_encoders(kind):
    """Build the models one voice action needs, BEFORE the caller takes a lock.

    All loaders are lru_cached, so this is exactly the work the action would do
    anyway — just earlier. NOT free, only better-placed: a cold load holds the
    GIL in 1.4-1.6 s blocks (measured), and every thread including the frame
    pump pauses with it. There is no way to pay that cost invisibly; the two
    call sites choose the least-bad moments. Two callers, two reasons:

    - The phone pointerdown handler calls it on a background thread, so
      Whisper's ~2.7 s and Nomic's ~4 s first loads happen while "LISTENING"
      is on screen instead of adding to the first teach of a session. (The
      laptop path deliberately does NOT: those GIL holds starve the local
      recorder's 100 ms reads and chop the audio.)
    - _process calls it (before transcribing, before the live-state lock) so a
      release that beats the warm simply waits on the lock here. Nomic's load
      paid inside the live-state lock stalls tracking long enough for
      DEAD_SECONDS to delete live tracks — the one way the app itself can
      cause the flicker it is trying not to have.

    `kind` is the same "t"/"a" that already flows through the live app. Teach
    never searches CLIP's text space, and this board budgets memory tightly
    enough that loading an encoder it will not use is worth avoiding.
    """
    with _warm_lock:
        _asr_model()         # every voice action transcribes first
        _text_model()        # teach stores a text vector; ask queries one
        if kind == "a":
            _clip_text()     # ask also searches the image space with CLIP text


def warm_up(progress=lambda name: None):
    """Load the one encoder the camera loop uses on every frame.

    Speech and text encoders are deliberately NOT loaded here. They are only
    needed while a human holds a button to teach or ask, so their first-use
    load hides under the utterance — and until then their ~1.7 GB stays free
    for the detector. Loading all four up front is what pushes an 8 GB Jetson
    into swap, and a swapping robot answers in seconds per frame.
    """
    progress("CLIP vision encoder")
    import numpy as np
    embed_crop(np.zeros((32, 32, 3), dtype=np.uint8))
