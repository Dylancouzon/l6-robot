"""The course's exact model stack: Nomic 768 (text), CLIP 512 (image), Whisper.

Same models, same FastEmbed/onnx-asr loaders as the course notebooks — the
"same stack" claim in L6 depends on this file staying aligned with
on-device-memory-course/.build/utils/{embeddings,audio}.py.
"""
from functools import lru_cache

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_DIM = 768
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
CLIP_DIM = 512
WHISPER_MODEL = "whisper-base"


@lru_cache(maxsize=1)
def _text_model():
    from fastembed import TextEmbedding
    return TextEmbedding(NOMIC_MODEL)


@lru_cache(maxsize=1)
def _clip_vision():
    from fastembed import ImageEmbedding
    return ImageEmbedding(CLIP_VISION_MODEL)


@lru_cache(maxsize=1)
def _clip_text():
    from fastembed import TextEmbedding
    return TextEmbedding(CLIP_TEXT_MODEL)


@lru_cache(maxsize=1)
def _asr_model():
    import onnx_asr
    return onnx_asr.load_model(WHISPER_MODEL, providers=["CPUExecutionProvider"])


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
    """Load every model once so the live loop never stalls mid-demo."""
    progress("Nomic text encoder")
    embed_text("warm up")
    progress("CLIP")
    embed_query_clip("warm up")
    import numpy as np
    embed_crop(np.zeros((32, 32, 3), dtype=np.uint8))
    progress("Whisper")
    _asr_model()
