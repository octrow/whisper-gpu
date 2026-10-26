"""GPU-error vs cache-miss classification, against real captured error strings.

No GPU and no model download here — these are pure string classifiers.
"""

from __future__ import annotations

import pytest

from whisper_gpu import is_cache_miss, is_gpu_error

# Real messages seen from CTranslate2 / faster-whisper / HF hub in the source projects.
GPU_ERRORS = [
    "CUDA failed with error out of memory",
    "CUDA failed with error invalid device ordinal",
    "cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED",
    "Library cudnn_ops_infer.so.9 is not found",
    "parallel_for failed: cudaErrorMemoryAllocation: out of memory",
    "cudaErrorNoDevice: no CUDA-capable device is detected",
]

NON_GPU_ERRORS = [
    "Invalid data found when processing input",
    "Failed to load audio: ffmpeg returned 1",
    "moov atom not found",
    "Unsupported subtitle format: /tmp/x.txt",
]


@pytest.mark.parametrize("msg", GPU_ERRORS)
def test_is_gpu_error_true(msg: str) -> None:
    assert is_gpu_error(RuntimeError(msg))


@pytest.mark.parametrize("msg", NON_GPU_ERRORS)
def test_is_gpu_error_false(msg: str) -> None:
    assert not is_gpu_error(RuntimeError(msg))


def test_is_gpu_error_is_case_insensitive() -> None:
    assert is_gpu_error(RuntimeError("OUT OF MEMORY"))


CACHE_MISSES = [
    (
        "We couldn't connect to huggingface.co to load this file, couldn't find it in the "
        "cached files and it looks like Systran/faster-whisper-large-v3 is not the path to a "
        "directory containing a file named config.json. local_files_only=True"
    ),
    "Entry Not Found for url: https://huggingface.co/.../model.bin",
    "No such file or directory: '/home/u/.cache/huggingface/hub/models--x/snapshots'",
    "Path /models/large-v3 does not exist",
]


@pytest.mark.parametrize("msg", CACHE_MISSES)
def test_is_cache_miss_true(msg: str) -> None:
    assert is_cache_miss(OSError(msg))


@pytest.mark.parametrize("msg", GPU_ERRORS)
def test_cuda_failure_is_never_a_cache_miss(msg: str) -> None:
    """The whole point of `is_cache_miss`: "cudnn ... is not found" contains
    "not found" but must NOT trigger a pointless 3 GB re-download."""
    assert not is_cache_miss(RuntimeError(msg))


def test_is_cache_miss_uses_exception_type_name() -> None:
    class EntryNotFoundError(Exception):
        pass

    assert is_cache_miss(EntryNotFoundError("404"))


def test_is_cache_miss_false_for_unrelated_error() -> None:
    assert not is_cache_miss(ValueError("beam_size must be positive"))
