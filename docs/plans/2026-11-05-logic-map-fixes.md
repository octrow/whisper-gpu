# Plan: logic map findings

Findings from writing `docs/logic-map.md` on 2026-11-05.
The map describes the code as it is; this file says what is wrong with it and what we propose. New findings are appended.
A fixed finding gets ✅ and the map is updated in the same commit.

Severity: HIGH: wrong result for the user. MEDIUM: worse result than it could be. LOW: noise, debt, docs.
"Question": not sure if bug or intent; owner decides. "Not confirmed": needs a live run.

## Findings

| # | Map step | What is wrong | We propose | Severity |
|---|---|---|---|---|
| 1 ✅ 2026-11-05 | 4 | `get_whisper` caches by device only (`transcribe.py:127`); `get_whisper("cuda", "tiny")` after `"large-v3"` returns large-v3 | key the cache by `(device, model, compute)` | MEDIUM |
| 2 | 4 | `is_cache_miss` markers `"cache"` and `"not found"` (`transcribe.py:69`, checked against "<Type>: <message>" at `:81`) match unrelated load errors (e.g. `PermissionError` on `~/.cache/huggingface`) and start a second load with `local_files_only=False` (`transcribe.py:151`): a Hub round-trip that can stall offline, a real download only if files are missing, and the first error survives only as `__context__`. Review 2026-11-05: "ffmpeg not found" cannot reach this check (audio is decoded during transcribe, not model load); severity lowered from MEDIUM, the load still fails loudly | narrow markers to HF/`local_files_only` exception types | LOW |
| 3 ✅ 2026-11-05 | 2 | `_join_dedup` (`subtitles.py:64`) compares whole entries; a rolling VTT cue with two lines joins into one entry and is never equal to the next. Review 2026-11-05 confirmed on webvtt-py 0.5.1: cues "hello there", "hello there / how are you", "how are you / fine thanks" give three lines with each phrase twice | split cue text on newlines before dedup | MEDIUM |
| 4 | 1 | `find_subtitle` uses the name up to the first dot (`subtitles.py:20`); ids with dots match wrong files; several languages: the alphabetical first wins | match `<name>.` prefix by stem; add optional `lang` preference | LOW |
| 5 | 3, 6 | `warn_if_low_vram` is never called by `transcribe_with_fallback`; no function runs captions → preflight → whisper. Review 2026-11-05: the `__init__` docstring (`__init__.py:3`) tells the consumer to cascade, it does not claim a built-in cascade; the README usage shows the consumer-side cascade | state in docs, or add a `transcribe_any` helper | LOW |
| 6 | 2 | SRT inline tags (`<i>`, `<b>`, `<font>`) are not stripped in `_parse_srt` (`subtitles.py:49`). Review 2026-11-05: `.vtt` is fine, webvtt-py 0.5.1 `Caption.text` drops `<c>`/`<i>` (checked on a sample file); `.srt` confirmed: `<i>Hello</i>` comes out as is | strip tags in `_parse_srt` | LOW |
| 7 | 6 | `free_gpu_model()` runs inside the `except` block (`transcribe.py:189-200`); `exc.__traceback__` keeps the `whisper_transcribe` frame and its `whisper` local alive, so the CUDA model is not freed until the function returns (checked with a stub model: freed only after return). The "release our ~3GB before CPU work" comment (`:196`) does not hold | leave the `except` block before the CPU retry (set a flag, retry after it), or clear the traceback | LOW |
| 8 ✅ 2026-11-05 keep as is | 6 | `transcribe_with_fallback` keeps no "GPU is dead" state. The comment at `transcribe.py:60-63` says a CUDA error corrupts the context and every later GPU call dies, yet the next call with `device="cuda"` preloads and loads large-v3 on `cuda` again, fails, and falls back, once per file. gce keeps its own `gpu_dead` flag (`gce/transcribe.py:402-405`) and does not use this path for that | document "pass `device_used` to the next call", or keep a module-level flag reset by `free_gpu_model` | MEDIUM. Decision 2026-11-05: keep as is, per-call retry is by design (a transient VRAM clash recovers on its own; restart the process for a stuck GPU); documented in the `transcribe_with_fallback` docstring and the comment above `GPU_ERR_MARKERS` |
| 9 | 2 | `_parse_srt` drops every line that is only digits (`subtitles.py:57`), not only the index line: a caption "1999" or "100" disappears, and an all-digit cue vanishes (checked: cue 2 "1999" gives no output) | drop only the first line of a block when it is digits | LOW |
| 10 ✅ 2026-11-05 | 2 | `.srt` is read as `utf-8`, not `utf-8-sig` (`subtitles.py:43`); a BOM makes the first index `"\ufeff1"`, which is not `isdigit()`, so the text starts with `"\ufeff1 "` (checked) | read with `encoding="utf-8-sig"` | LOW |
| 11 | 6 | `is_gpu_error` matches markers against the whole message (`transcribe.py:75`); a decode error that quotes an audio path containing "cuda" (PyAV puts the file name in the message) counts as a GPU error: the CUDA model is dropped and the CPU retry fails with the same error | match on the message without the file path, or on exception types | LOW |

## Order of work

1. #1, #8, #7 (model cache and fallback, `tests/test_fallback.py`), then #2 and #11 (classifiers, `tests/test_classifiers.py`).
2. #3, #4, #6, #9, #10 (subtitles, `tests/test_subtitles.py`).

## Done when

- [ ] every finding has ✅ or a "keep as is" decision with a reason;
- [ ] Known gaps in the map lists every open HIGH and MEDIUM finding.
