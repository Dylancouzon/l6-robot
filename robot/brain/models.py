"""Lazy loaders for the Nomic, CLIP, and Whisper models.

Changing an encoder requires recalibrating with `testdata/verify_scores.py`.
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

# Leave CPU capacity for the camera and rendering threads.
ENCODER_THREADS = 2

# Whisper is user-facing and does not run beside another encoder.
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


# Set another Whisper language code to teach in that language, or None to
# enable automatic detection at the cost of another inference pass.
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
    """Load the speech and text models before a voice action takes a lock."""
    with _warm_lock:
        _asr_model()         # every voice action transcribes first
        _text_model()        # teach stores a text vector; ask queries one


def warm_text():
    """Load only the text encoder, using the shared model-loading lock."""
    with _warm_lock:
        _text_model()


def warm_up(progress=lambda name: None):
    """Load the image encoder used by the camera loop."""
    progress("CLIP vision encoder")
    import numpy as np
    embed_crop(np.zeros((32, 32, 3), dtype=np.uint8))
