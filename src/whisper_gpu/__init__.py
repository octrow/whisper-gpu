"""whisper-gpu — GPU transcription primitives + native caption parsing.

Cascade the two the way the source projects do: try native captions first (free,
no GPU), fall back to faster-whisper with an automatic GPU→CPU retry.

``faster_whisper`` is imported lazily inside the functions, so importing this
package on a machine without GPU deps is safe; ``whisper_gpu.subtitles`` needs
only ``webvtt-py``.
"""

from .subtitles import (
    find_subtitle,
    lang_from_subtitle_name,
    parse_subtitle,
)
from .transcribe import (
    CACHE_MISS_MARKERS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL,
    GPU_ERR_MARKERS,
    WHISPER_MIN_FREE_MIB,
    free_gpu_model,
    get_whisper,
    gpu_free_mib,
    is_cache_miss,
    is_gpu_error,
    preload_cuda_libs,
    transcribe_with_fallback,
    warn_if_low_vram,
    whisper_transcribe,
)

__all__ = [
    "CACHE_MISS_MARKERS",
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_MODEL",
    "GPU_ERR_MARKERS",
    "WHISPER_MIN_FREE_MIB",
    "find_subtitle",
    "free_gpu_model",
    "get_whisper",
    "gpu_free_mib",
    "is_cache_miss",
    "is_gpu_error",
    "lang_from_subtitle_name",
    "parse_subtitle",
    "preload_cuda_libs",
    "transcribe_with_fallback",
    "warn_if_low_vram",
    "whisper_transcribe",
]
