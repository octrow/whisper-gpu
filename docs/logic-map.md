# Logic map

How an audio file becomes plain text: native captions first, faster-whisper on GPU, then on CPU.
The map describes the code as it is today, step by step.
Each step says what it decides, where the flow branches, and which module does it.

Known gaps and fixes in progress are in `docs/plans/2026-11-05-logic-map-fixes.md`.

Rule: if you change a step or a branch, update this file in the same commit.
`tests/test_logic_map.py` checks that every `Module:` line points to a real module.

whisper-gpu is a library of primitives. No function runs the whole cascade; the consumer (tlf, gce) calls the steps in this order.

## The flow at a glance

```
audio file (+ sibling caption file the downloader may have written)
  │
  ├─ 1  Find caption ........... is there a <id>.<lang>.vtt/.srt next to the audio?
  ├─ 2  Parse caption .......... caption file -> deduplicated plain text (no GPU)
  │        └─ done if a caption was found
  ├─ 3  Preflight VRAM ......... is enough GPU memory free? (advisory only)
  ├─ 4  Get model .............. cached WhisperModel per device; local cache or download
  ├─ 5  Transcribe ............. faster-whisper on the device -> (text, lang)
  └─ 6  GPU -> CPU fallback .... did a GPU error happen? redo on CPU
  │
(text, lang) or (text, lang, device_used)
```

Steps 1-2 never import faster-whisper. Steps 3-6 never import webvtt.
Step 3 is not called by steps 4-6; the consumer calls it.

## Where the flow splits by failure kind

The same exception is classified by substring markers in `transcribe`.

| Kind | Switched on by | Steps it changes | What changes |
|---|---|---|---|
| GPU error | `GPU_ERR_MARKERS` in lowercased message ("out of memory", "cuda", "cublas", "cudnn", "device ordinal", "parallel_for", "cudaerror") | 6 | retry on CPU |
| Cache miss | `CACHE_MISS_MARKERS` ("local_files_only", "not found", "no such file", "does not exist", "cache", "snapshot", "entrynotfound") and no "cuda"/"cudnn"/"cublas", both checked in lowercased "<ExceptionType>: <message>" | 4 | load again with download allowed |
| Anything else | neither | 4, 5 | exception propagates |

Examples:
- "CUDA out of memory" during transcribe on `cuda` → GPU error → CPU retry.
- "cudnn library not found" during model load → not a cache miss (cuda words rule it out) → raised by step 4; under `transcribe_with_fallback` it is then a GPU error → CPU retry.
- Undecodable audio → neither → raised; the caller skip-lists it. The markers are matched against the whole message, so an error that quotes an audio path containing "cuda" counts as a GPU error (#11).

Module: `transcribe`

## Entry points

All entry points are public functions of the package (`__init__` re-exports). There is no CLI.

| Entry | Device | Fallback | Notes |
|---|---|---|---|
| `find_subtitle`, `parse_subtitle`, `lang_from_subtitle_name` | none | none | caption path only |
| `whisper_transcribe` | `cuda` | none | returns `(text, lang)`; model `large-v3`, `float16`, beam 5; no VAD filter |
| `transcribe_with_fallback` | `cuda` | **CPU on GPU error** (`cpu_fallback=True`) | returns `(text, lang, device_used)`; CPU run uses `int8` and ignores `compute_type`; keeps no memory of the failure, the next call tries `cuda` again (#8) |
| `get_whisper`, `warn_if_low_vram`, `gpu_free_mib`, `free_gpu_model`, `preload_cuda_libs`, `is_gpu_error`, `is_cache_miss` | - | - | building blocks; constants `DEFAULT_MODEL`, `DEFAULT_COMPUTE_TYPE`, `WHISPER_MIN_FREE_MIB`, both marker tuples are exported too |

Legacy underscore aliases (`_get_whisper`, `_is_gpu_error`, ...) exist in `transcribe` for tlf/gce; they are not re-exported by `__init__`.

Module: `transcribe`, `subtitles`

## Steps

### 1. Find caption

Decides: is there a caption file next to the audio.
In: audio `Path`. Out: caption `Path` or `None`.
Branches:
- sibling `<id>.*` with suffix `.vtt` or `.srt` (case-insensitive) → first in sorted order wins;
- none → `None`, the consumer goes to step 3.
`<id>` is the audio file name up to its first dot.
Fails: never raises; no match is `None`.
Module: `subtitles`

### 2. Parse caption

Decides: how a caption file turns into text.
In: caption `Path`. Out: `str`, one line per distinct caption.
Branches:
- `.vtt` → `webvtt` library, one entry per cue line; `Caption.text` already drops inline tags (`<c>`, `<i>`; checked on webvtt-py 0.5.1); a multi-line cue is split into lines before dedup (#3 fixed);
- `.srt` → own parser: split on blank lines, drop every line that is only digits (not just the index: a "1999" caption goes too, #9) and every `-->` line, join the rest of the block with spaces; inline tags are kept (#6); the file is read as utf-8-sig so a BOM is dropped (#10 fixed);
- other suffix → `ValueError`.
Dedup: whitespace collapsed per entry; an entry equal to the previous one is dropped.
Language: `parse_subtitle` returns text only; the consumer calls `lang_from_subtitle_name`, which takes the second-to-last dot part of the name ("123.en.vtt" → "en"); fewer than 3 parts → "".
Fails: `.vtt` with webvtt-py missing → `ImportError`; bad file → webvtt/IO error raised; `.srt` read with `errors="replace"`.
Module: `subtitles`

### 3. Preflight VRAM

Decides: whether to warn about low free VRAM.
In: model name (used only in the message), threshold (3500 MiB). Out: free MiB or `None`.
Branches:
- `nvidia-smi` reads, first line (GPU 0 only) below threshold → log warning, never blocks;
- `nvidia-smi` missing, timed out (10 s) or unparsable (empty output on a failing exit code) → `None`, no warning.
Fails: fail-open; every exception is logged at debug and becomes `None`.
Module: `transcribe`

### 4. Get model

Decides: which `WhisperModel` serves a device, and whether to download.
In: device, model, compute type. Out: cached `WhisperModel`.
Branches:
- device already in cache → returned as is, model and compute type args are ignored;
- `cuda` → `preload_cuda_libs` first (ctypes load of pip cuBLAS, then cuDNN, `RTLD_GLOBAL`; no-op without the `nvidia` package);
- compute type: given value on `cuda`, `int8` on any other device;
- load with `local_files_only=True`; on a cache miss (see split table) load again with `local_files_only=False` (download).
No GPU: `cpu` path needs no preload.
Fails: any non-cache-miss exception is raised; a failed download raises (the first error is only its `__context__`). `faster_whisper` missing → `ImportError` before the load.
Module: `transcribe`

### 5. Transcribe

Decides: text and language for one file.
In: audio `Path`, device, model, beam size, compute type. Out: `(text, lang)`.
Text is the stripped, non-empty segment texts joined by newlines. `lang` is the detected language or "".
Segments are a lazy generator, so decode and CUDA errors surface while the text is joined, still inside this step.
Fails: raises whatever step 4 or faster-whisper raises. No non-empty segment gives `("", lang)`; no VAD filter is passed, so silence can still come back as hallucinated text.
Module: `transcribe`

### 6. GPU → CPU fallback

Decides: whether a failed GPU run is retried on CPU.
In: same as step 5 plus `cpu_fallback`. Out: `(text, lang, device_used)`.
Branches:
- success → device as requested;
- failure, `device == "cuda"`, `cpu_fallback`, GPU error marker → log warning, `free_gpu_model` (drops the `cuda` cache entry, `gc.collect`), run step 5 on `cpu`, return `"cpu"`;
- any other failure (CPU device, flag off, non-GPU error) → raised unchanged.
The failed model is still referenced by the live traceback during the CPU retry, so its VRAM is freed only after the function returns (#7).
No GPU: the default `device="cuda"` fails the load on every call (preload is a no-op, the CUDA error is a GPU error) and each call falls back to CPU.
Fails: a failure of the CPU retry is raised with no further fallback.
Module: `transcribe`

### Output

`(text, lang)` or `(text, lang, device_used)`; plain strings, no result type.
Module: `transcribe`

## Known gaps

- The `cache` and `not found` markers match any load error mentioning them, which starts a second load with Hub access (#2).
- `find_subtitle` picks the alphabetically first caption and breaks on ids containing dots (#4).
- `transcribe_with_fallback` keeps no "GPU is dead" state: every later call reloads the model on `cuda` and fails again (#8).
- The CUDA model's VRAM is held through the CPU retry (#7).
- SRT parsing drops all-digit captions and keeps inline tags (#9, #6).
