# API reference

Moved here from the README. For the step-by-step flow see [logic-map.md](logic-map.md).

## Why each piece exists

Each piece fixes a failure seen in practice.

* **cuDNN/cuBLAS preload.** CTranslate2 4.x loads cuDNN 9 with `dlopen` when the model loads, but it
  does not look in the lib dirs of the `nvidia-*-cu12` pip packages. Without `preload_cuda_libs()` the
  GPU model load hangs with no error.
* **`local_files_only=True` first.** The Hugging Face Hub check on every load can stall for minutes on a
  slow network. The model is fetched from the Hub only after a local load fails with a cache miss.
* **Cache miss or CUDA error.** `"Library cudnn_ops_infer.so.9 is not found"` contains `"not found"`.
  Treat it as a cache miss and you start a 3 GB re-download that fails anyway. `is_cache_miss()` rules
  out CUDA, cuDNN and cuBLAS before it looks for cache-miss markers.
* **GPU OOM, then CPU.** A CUDA OOM, usually because another process holds VRAM, corrupts the device
  context, and later calls die with "invalid device ordinal". The library detects it, drops its own CUDA
  model and redoes the file on CPU in `int8`, so good audio is not marked undecodable.

How the classifiers sort a few of the real error strings from `tests/test_classifiers.py`:

| Error message | `is_gpu_error` | `is_cache_miss` |
|---|---|---|
| `CUDA failed with error out of memory` | yes | no |
| `Library cudnn_ops_infer.so.9 is not found` | yes | no |
| `Path /models/large-v3 does not exist` | no | yes |
| `Invalid data found when processing input` | no | no |

## Lower-level pieces

Use these if you drive the loop yourself:

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

Models are cached per device in a module-level dict, so repeated calls reuse the loaded model.
`free_gpu_model()` drops the CUDA one and runs the garbage collector to release its ~3 GB.

## Functions

| | |
|---|---|
| `preload_cuda_libs()` | RTLD_GLOBAL-load pip cuBLAS, then cuDNN; no-op without them |
| `is_gpu_error(exc) -> bool` | CUDA/cuDNN/VRAM failure (transient) vs bad input |
| `is_cache_miss(exc) -> bool` | model not (fully) downloaded, the only case a download fixes |
| `gpu_free_mib() -> int \| None` | free VRAM via `nvidia-smi` |
| `warn_if_low_vram(model, min_free_mib) -> int \| None` | preflight warning, never blocks |
| `free_gpu_model()` | drop the cached CUDA model + GC |
| `get_whisper(device, model, compute_type)` | lazy, cached `WhisperModel` (CPU forced to `int8`) |
| `whisper_transcribe(audio_path, device, model, beam_size, compute_type) -> (text, lang)` | one transcribe call |
| `transcribe_with_fallback(..., cpu_fallback=True) -> (text, lang, device_used)` | cuda → cpu on GPU error |
| `find_subtitle / parse_subtitle / lang_from_subtitle_name` | VTT/SRT → deduped plain text |

Constants: `DEFAULT_MODEL="large-v3"`, `DEFAULT_COMPUTE_TYPE="float16"`,
`WHISPER_MIN_FREE_MIB=3500`, `GPU_ERR_MARKERS`, `CACHE_MISS_MARKERS`.

## Old names

The old underscore names from the source modules (`_whisper_transcribe`, `_is_gpu_error`,
`_get_whisper`, …) remain as aliases in `whisper_gpu.transcribe`, so moving existing code over is an
import swap.
