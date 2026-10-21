"""GPU Whisper transcription primitives.

The load-bearing plumbing, extracted byte-for-byte-equivalent from two repos
(tiktok-lazy-follower ``tlf/transcribe.py`` and get_cool_event ``gce/transcribe.py``):

* the cuDNN9/cuBLAS ctypes preload CTranslate2 needs to find pip-installed CUDA libs,
* a lazily-built, per-device ``WhisperModel`` cache,
* GPU-error / cache-miss classifiers (so a CUDA failure never triggers a 3 GB
  re-download, and a genuine cache miss does),
* a free-VRAM preflight probe,
* one transcribe call, plus an opt-in GPU→CPU fallback wrapper.

No config objects, no filesystem layout, no orchestration loop — those stay in the
consuming project. ``faster_whisper`` is imported lazily inside the functions, so
this module imports cleanly on a machine with no GPU deps installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("whisper_gpu.transcribe")

# Lazily-loaded whisper models, keyed by device ('cuda'/'cpu') — heavy import + load.
_WHISPER_MODELS: dict = {}

DEFAULT_MODEL = "large-v3"
DEFAULT_COMPUTE_TYPE = "float16"

# large-v3 needs roughly this much free VRAM to load + run; below it, expect OOM.
WHISPER_MIN_FREE_MIB = 3500


def preload_cuda_libs() -> None:
    """Preload pip-installed cuDNN/cuBLAS so CTranslate2 (faster-whisper backend)
    can find them. CTranslate2 4.x dlopen's cuDNN 9 at model-load time but does NOT
    search the `nvidia-*-cu12` pip packages' lib dirs — without this, GPU model load
    hangs silently. Loading the .so files RTLD_GLOBAL puts their symbols where
    CTranslate2's dlopen resolves them. No-op if the libs/GPU aren't present."""
    import ctypes
    import glob

    try:
        import nvidia
    except ImportError:
        return
    # `nvidia` is a namespace package — no __file__, so use __path__.
    bases = [Path(p) for p in getattr(nvidia, "__path__", [])]
    # order matters: cuBLAS first (cuDNN depends on it)
    for sub in ("cublas", "cudnn"):
        for base in bases:
            for so in sorted(glob.glob(str(base / sub / "lib" / "lib*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    log.debug("could not preload CUDA lib %s", so, exc_info=True)


# A CUDA error (usually out-of-memory because another process holds VRAM) corrupts
# the device context — every later GPU call then dies with "invalid device ordinal".
# Detect it so we (a) don't mislabel good inputs as "undecodable" and (b) stop
# hammering the dead GPU and switch to CPU.
GPU_ERR_MARKERS = ("out of memory", "cuda", "cublas", "cudnn",
                   "device ordinal", "parallel_for", "cudaerror")

# Markers of "the model isn't (fully) in the local HF cache" — the only failure a
# download can fix.
CACHE_MISS_MARKERS = ("local_files_only", "not found", "no such file",
                      "does not exist", "cache", "snapshot", "entrynotfound")


def is_gpu_error(exc: Exception) -> bool:
    """True when `exc` looks like a CUDA/cuDNN/VRAM failure rather than bad input."""
    return any(m in str(exc).lower() for m in GPU_ERR_MARKERS)


def is_cache_miss(exc: Exception) -> bool:
    """True when the local_files_only load failed because the model isn't (fully)
    downloaded — the only case a download can fix. CUDA/cuDNN errors re-raise."""
    s = f"{type(exc).__name__}: {exc}".lower()
    # A CUDA/cuDNN/cuBLAS failure is NOT a cache miss — even though "cudnn library not
    # found" contains "not found". Rule these out before the cache-miss markers.
    if any(m in s for m in ("cuda", "cudnn", "cublas")):
        return False
    return any(m in s for m in CACHE_MISS_MARKERS)


def gpu_free_mib() -> int | None:
    """Free VRAM in MiB via nvidia-smi, or None if it can't be read. Used to warn
    up front when another process is holding the GPU (the usual OOM cause)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return int(out.stdout.split("\n")[0].strip())
    except Exception:
        log.debug("nvidia-smi free-VRAM query failed", exc_info=True)
        return None


def warn_if_low_vram(model: str = DEFAULT_MODEL,
                     min_free_mib: int = WHISPER_MIN_FREE_MIB) -> int | None:
    """Preflight VRAM check: log a warning (never block) when another process is
    holding the GPU. Returns the free MiB, or None when unreadable."""
    free = gpu_free_mib()
    if free is not None and free < min_free_mib:
        log.warning("Only %d MiB GPU free (need ~%d for %s). Another process is likely "
                    "holding VRAM — check `nvidia-smi` and free it, or whisper will fall "
                    "back to CPU.", free, min_free_mib, model)
    return free


def free_gpu_model() -> None:
    """Drop our cached CUDA model and run GC so its VRAM is released before a CPU
    fallback — and so a later re-run starts clean."""
    import gc
    _WHISPER_MODELS.pop("cuda", None)
    gc.collect()


def get_whisper(device: str, model: str = DEFAULT_MODEL,
                compute_type: str = DEFAULT_COMPUTE_TYPE):
    """Lazily build (and cache) the whisper model for `device` ('cuda' or 'cpu')."""
    if device not in _WHISPER_MODELS:
        if device == "cuda":
            preload_cuda_libs()
        from faster_whisper import WhisperModel

        # int8 on CPU — far faster than float16 and accurate enough for transcripts.
        compute = compute_type if device == "cuda" else "int8"
        log.info("Loading faster-whisper %s on %s…", model, device)
        try:
            # local_files_only: use the already-downloaded HF cache and SKIP the
            # per-load network check to the HF Hub — that check can hang for minutes
            # when the network is slow/unreachable (the "unauthenticated requests to
            # the HF Hub" warning + stall).
            _WHISPER_MODELS[device] = WhisperModel(
                model, device=device, compute_type=compute,
                local_files_only=True,
            )
        except Exception as exc:
            # Only a genuine cache miss can be fixed by downloading. A CUDA/cuDNN
            # failure would trigger a pointless 3GB re-download and then fail again.
            if not is_cache_miss(exc):
                raise
            log.info("Model not in local cache — downloading %s from HF Hub (one-time)…",
                     model)
            _WHISPER_MODELS[device] = WhisperModel(
                model, device=device, compute_type=compute,
                local_files_only=False,
            )
        log.info("Whisper model ready (%s).", device)
    return _WHISPER_MODELS[device]


def whisper_transcribe(
    audio_path: Path, device: str = "cuda", model: str = DEFAULT_MODEL,
    beam_size: int = 5, compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> tuple[str, str]:
    """Return (text, detected_lang) for `audio_path` using faster-whisper on `device`."""
    whisper = get_whisper(device, model, compute_type)
    segments, info = whisper.transcribe(str(audio_path), beam_size=beam_size)
    text = "\n".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text, info.language or ""


def transcribe_with_fallback(
    audio_path: Path, device: str = "cuda", model: str = DEFAULT_MODEL,
    beam_size: int = 5, compute_type: str = DEFAULT_COMPUTE_TYPE,
    cpu_fallback: bool = True,
) -> tuple[str, str, str]:
    """Transcribe on `device`, retrying on CPU when the GPU dies.

    Returns ``(text, lang, device_used)``. A CUDA failure (usually OOM because
    another process holds VRAM) corrupts the device context, so we release our own
    model first and then redo the file on CPU instead of mislabeling good input as
    undecodable. Non-GPU errors (undecodable audio) propagate unchanged — the caller
    decides whether to skip-list them.
    """
    try:
        text, lang = whisper_transcribe(
            audio_path, device=device, model=model, beam_size=beam_size,
            compute_type=compute_type,
        )
        return text, lang, device
    except Exception as exc:
        if not (device == "cuda" and cpu_fallback and is_gpu_error(exc)):
            raise
        log.warning("GPU transcription failed (%s). Likely VRAM is occupied (check "
                    "`nvidia-smi` for another process). Falling back to CPU — slower; "
                    "free VRAM and re-run for GPU speed.",
                    str(exc).splitlines()[0] or type(exc).__name__)
        free_gpu_model()  # release our ~3GB before CPU work
        text, lang = whisper_transcribe(
            audio_path, device="cpu", model=model, beam_size=beam_size,
        )
        return text, lang, "cpu"


# --- Legacy private aliases -------------------------------------------------
# The extracted sources called these with a leading underscore; keep the old names
# working so migrating tlf/gce is a pure import swap.
_preload_cuda_libs = preload_cuda_libs
_is_gpu_error = is_gpu_error
_is_cache_miss = is_cache_miss
_gpu_free_mib = gpu_free_mib
_free_gpu_model = free_gpu_model
_get_whisper = get_whisper
_whisper_transcribe = whisper_transcribe
_GPU_ERR_MARKERS = GPU_ERR_MARKERS
_WHISPER_MIN_FREE_MIB = WHISPER_MIN_FREE_MIB
