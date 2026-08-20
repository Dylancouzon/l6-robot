"""The course's model stack: Nomic 768 (text), CLIP 512 (image), Whisper (speech).

Same models and the same FastEmbed and onnx-asr loaders as the course
notebooks, on the versions pinned in pyproject.toml. The recognition threshold
is calibrated against these exact encoders and the crop pipeline, so changing
one means re-running testdata/verify_scores.py.

Every loader is cached and lazy: a model is built once, the first time
something asks for it. That is what lets an 8 GB Jetson run this (see warm_up).
"""
import os
import threading
from functools import lru_cache
from pathlib import Path

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_DIM = 768
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_DIM = 512
WHISPER_MODEL = "whisper-base"
# CLIP's text tower is deliberately not here. Searching the image space with
# the words of a question was measured useless, so recall picks the object in
# Nomic's text space instead (see Memory.best_taught).

# ONNX Runtime gives every session a thread per core and lets idle ones spin,
# so the encoders the detect thread calls would starve the camera loop. Two
# threads each leaves it room to keep drawing.
ENCODER_THREADS = 2

# Whisper gets more: it is the one model a human waits on, and the only one
# that never runs beside another encoder. Do not raise past 4 without
# re-measuring the feed - the board has six cores and the frame pump needs one.
ASR_THREADS = 4

# FastEmbed caches ONNX files in /tmp by default, and systemd-tmpfiles prunes
# /tmp at 30 days. That is 1.1 GB of encoders a robot on its own hotspot can
# never fetch again, so keep them somewhere durable.
CACHE_DIR = (os.environ.get("FASTEMBED_CACHE_PATH")
             or str(Path.home() / ".cache" / "fastembed"))


def _sess_options():
    """ONNX Runtime's threading controls, for onnx-asr."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ASR_THREADS
    opts.inter_op_num_threads = ASR_THREADS
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
def _asr_model():
    import onnx_asr
    return onnx_asr.load_model(WHISPER_MODEL, sess_options=_sess_options(),
                               providers=["CPUExecutionProvider"])


def embed_text(text):
    """Embed one transcript for storage. Nomic, 768-d."""
    return next(_text_model().embed([text])).tolist()


def embed_query(text):
    """Embed one question. Nomic's query prefix matters for retrieval."""
    return next(_text_model().query_embed([text])).tolist()


def embed_crop(bgr):
    """Embed one OpenCV BGR crop with CLIP's vision tower. 512-d."""
    from PIL import Image
    img = Image.fromarray(bgr[:, :, ::-1])
    return next(_clip_vision().embed([img])).tolist()


# Naming the language halves every voice action. This Whisper export holds the
# encoder and decoder in one graph, so letting it guess runs that whole graph a
# second time. It also stops a stray guess storing an English teach under a
# Cyrillic name. Set another Whisper language code to teach in that language,
# or None to pay the second pass.
LANGUAGE = "en"


def transcribe(wav_path):
    """Local Whisper speech-to-text on one WAV file."""
    kwargs = {"language": LANGUAGE} if LANGUAGE else {}
    return _asr_model().recognize(wav_path, **kwargs).strip()


# warm_encoders runs on two threads per interaction (a background warm while
# the button is held, and again before the action transcribes). lru_cache does
# not serialize a cache miss, so without this lock both would build the same
# model at once.
_warm_lock = threading.Lock()


def warm_encoders():
    """Build the models a voice action needs, before the caller takes a lock.

    Everything here is lru_cached, so this is the work the action would do
    anyway, moved earlier. It is not free: a cold load blocks every thread for
    a second or two, so the two call sites pick the least-bad moments - under
    the button hold, and before the app's live-state lock is claimed.
    """
    with _warm_lock:
        _asr_model()         # every voice action transcribes first
        _text_model()        # teach stores a text vector; ask queries one


def warm_text():
    """Build ONLY the text encoder, under the same lock, for a caller that
    embeds without transcribing.

    The rename path needs Nomic and nothing else; dragging Whisper's ~2.7 s in
    behind it (as `warm_encoders` would) is load a typed action never uses. The
    lock is the point: `embed_text` takes none of its own, so a cold build
    started from an HTTP thread while a pointerdown warm ran would build Nomic
    twice at once - ~1 GB of transient extra on a board with none to spare.
    """
    with _warm_lock:
        _text_model()


def warm_up(progress=lambda name: None):
    """Load the one encoder the camera loop uses on every frame.

    Speech and text encoders are deliberately left out: they are only needed
    while a human holds a button, so their first load hides under the
    utterance and until then their ~1.7 GB stays free for the detector.
    Loading all four up front is what pushes an 8 GB Jetson into swap.
    """
    progress("CLIP vision encoder")
    import numpy as np
    embed_crop(np.zeros((32, 32, 3), dtype=np.uint8))
