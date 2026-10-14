# whisper-gpu-kit

Installed as `whisper-gpu-kit`, imported as `whisper_gpu`.

GPU transcription plumbing for `faster-whisper`, extracted from two repos that had
the same code byte-for-byte (`tiktok-lazy-follower/src/tlf/transcribe.py`,
`get_cool_event/src/gce/transcribe.py`) plus the companion caption parser
(`tlf/subtitles.py`).

Python ≥3.12. `faster_whisper` is imported **lazily inside the functions**, so the
package imports fine on a machine with no GPU deps; `whisper_gpu.subtitles` needs
only `webvtt-py`.

```bash
uv add whisper-gpu-kit                      # + `[gpu]` extra for the CUDA runtimes
```

## Why it exists

* **cuDNN/cuBLAS preload.** CTranslate2 4.x `dlopen`s cuDNN 9 at model-load time
  but does not search the `nvidia-*-cu12` pip packages' lib dirs — without
  `preload_cuda_libs()` the GPU model load hangs *silently*.
* **`local_files_only=True` first.** The per-load HF Hub check can stall for
  minutes on a slow network.
* **Cache-miss vs CUDA-error disambiguation.** `"cudnn ... is not found"` contains
  `"not found"`; treating it as a cache miss triggers a pointless 3 GB re-download
  that then fails anyway. `is_cache_miss()` rules CUDA out first.
* **GPU-OOM → CPU fallback.** A CUDA OOM (usually another process holding VRAM)
  corrupts the device context; every later call dies with "invalid device ordinal".
  Detect it, release our model, redo on CPU (`int8`) instead of mislabeling good
  audio as undecodable.

## Usage

Captions first (free, no GPU), whisper second:

```python
from pathlib import Path
from whisper_gpu import (find_subtitle, parse_subtitle, lang_from_subtitle_name,
                         transcribe_with_fallback, warn_if_low_vram)

audio = Path("data/audio/someone/7123456789.m4a")

sub = find_subtitle(audio)                  # sibling <id>.<lang>.vtt/.srt
if sub is not None:
    text, lang, source = parse_subtitle(sub), lang_from_subtitle_name(sub), "native_sub"
else:
    warn_if_low_vram()                      # logs if another process holds the GPU
    text, lang, device = transcribe_with_fallback(audio)   # cuda → cpu on GPU failure
```

Lower-level pieces, if you drive the loop yourself:

```python
from whisper_gpu import whisper_transcribe, is_gpu_error, free_gpu_model

try:
    text, lang = whisper_transcribe(audio, device="cuda", beam_size=5)
except Exception as exc:
    if is_gpu_error(exc):                   # transient: VRAM contention → retry later
        free_gpu_model()
        text, lang = whisper_transcribe(audio, device="cpu")
    else:
        raise                               # permanent: undecodable audio → skip-list it
```

Models are cached per device in a module-level dict, so repeated calls reuse the
loaded model; `free_gpu_model()` drops the CUDA one and GCs to release its ~3 GB.

## API

| | |
|---|---|
| `preload_cuda_libs()` | RTLD_GLOBAL-load pip cuBLAS then cuDNN; no-op without them |
| `is_gpu_error(exc) -> bool` | CUDA/cuDNN/VRAM failure (transient) vs bad input |
| `is_cache_miss(exc) -> bool` | model not (fully) downloaded — the only case a download fixes |
| `gpu_free_mib() -> int \| None` | free VRAM via `nvidia-smi` |
| `warn_if_low_vram(model, min_free_mib) -> int \| None` | preflight warning, never blocks |
| `free_gpu_model()` | drop the cached CUDA model + GC |
| `get_whisper(device, model, compute_type)` | lazy, cached `WhisperModel` (CPU forced to `int8`) |
| `whisper_transcribe(audio_path, device, model, beam_size, compute_type) -> (text, lang)` | one transcribe call |
| `transcribe_with_fallback(..., cpu_fallback=True) -> (text, lang, device_used)` | cuda → cpu on GPU error |
| `find_subtitle / parse_subtitle / lang_from_subtitle_name` | VTT/SRT → deduped plain text |

Constants: `DEFAULT_MODEL="large-v3"`, `DEFAULT_COMPUTE_TYPE="float16"`,
`WHISPER_MIN_FREE_MIB=3500`, `GPU_ERR_MARKERS`, `CACHE_MISS_MARKERS`.

The old underscore names from the source modules (`_whisper_transcribe`,
`_is_gpu_error`, `_get_whisper`, …) remain as aliases in `whisper_gpu.transcribe`
so migrating the two consumers is an import swap.

## Tests

```bash
uv run --package whisper-gpu --extra dev pytest whisper-gpu/tests -q
```

No GPU and no model download: the classifiers are pure string logic, and the
model layer is monkeypatched.
