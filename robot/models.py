"""The course's exact model stack: Nomic 768 (text), CLIP 512 (image), Whisper.

Same models and the same FastEmbed and onnx-asr loaders as the course
notebooks. The "same stack" claim in L6 depends on this file staying aligned
with the course's helper.py, and on the pinned versions in pyproject.toml:
the 0.80 recognition threshold is calibrated against those exact encoders.

Each loader is cached, so a model is built once and only when something first
asks for it. That laziness is what lets an 8 GB Jetson run this: see warm_up.
"""
from functools import lru_cache

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
    return TextEmbedding(NOMIC_MODEL, threads=ENCODER_THREADS)


@lru_cache(maxsize=1)
def _clip_vision():
    from fastembed import ImageEmbedding
    return ImageEmbedding(CLIP_VISION_MODEL, threads=ENCODER_THREADS)


@lru_cache(maxsize=1)
def _clip_text():
    from fastembed import TextEmbedding
    return TextEmbedding(CLIP_TEXT_MODEL, threads=ENCODER_THREADS)


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
