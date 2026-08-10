"""Fallback + model-cache logic with the faster-whisper layer monkeypatched out.

No GPU, no CUDA libs, no model download.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from whisper_gpu import transcribe as T


class _Seg:
    def __init__(self, text: str) -> None:
        self.text = text


class _Info:
    def __init__(self, language: str | None) -> None:
        self.language = language


class _FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, model: str, device: str, compute_type: str,
                 local_files_only: bool) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.local_files_only = local_files_only
        self.calls: list[tuple[str, int]] = []

    def transcribe(self, path: str, beam_size: int = 5):
        self.calls.append((path, beam_size))
        return [_Seg(" hi "), _Seg("  "), _Seg("there")], _Info("en")


@pytest.fixture(autouse=True)
def _clear_model_cache():
    T._WHISPER_MODELS.clear()
    yield
    T._WHISPER_MODELS.clear()


@pytest.fixture
def fake_whisper(monkeypatch):
    """Install a fake `faster_whisper` module and neutralize the CUDA preload."""
    built: list[_FakeModel] = []
    fail_with: dict[str, Exception] = {}

    def factory(model, device, compute_type, local_files_only):
        exc = fail_with.get("load")
        if exc is not None:
            raise exc
        m = _FakeModel(model, device, compute_type, local_files_only)
        built.append(m)
        return m

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setattr(T, "preload_cuda_libs", lambda: None)
    return types.SimpleNamespace(built=built, fail_with=fail_with)


def test_whisper_transcribe_joins_nonempty_segments(fake_whisper) -> None:
    text, lang = T.whisper_transcribe(Path("/tmp/a.m4a"), device="cuda")
    assert text == "hi\nthere"
    assert lang == "en"
    assert fake_whisper.built[0].device == "cuda"
    assert fake_whisper.built[0].calls == [("/tmp/a.m4a", 5)]


def test_cpu_device_forces_int8_compute(fake_whisper) -> None:
    T.get_whisper("cpu", "large-v3", compute_type="float16")
    assert fake_whisper.built[0].compute_type == "int8"


def test_cuda_keeps_requested_compute_type(fake_whisper) -> None:
    T.get_whisper("cuda", "large-v3", compute_type="float16")
    assert fake_whisper.built[0].compute_type == "float16"


def test_model_is_built_once_per_device(fake_whisper) -> None:
    a = T.get_whisper("cuda")
    b = T.get_whisper("cuda")
    assert a is b
    assert len(fake_whisper.built) == 1
    T.get_whisper("cpu")
    assert len(fake_whisper.built) == 2


def test_load_prefers_local_files_only(fake_whisper) -> None:
    T.get_whisper("cuda")
    assert fake_whisper.built[0].local_files_only is True


def test_cache_miss_retries_with_download(monkeypatch) -> None:
    attempts: list[bool] = []

    def factory(model, device, compute_type, local_files_only):
        attempts.append(local_files_only)
        if local_files_only:
            raise OSError("... local_files_only=True and config.json was not found")
        return _FakeModel(model, device, compute_type, local_files_only)

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setattr(T, "preload_cuda_libs", lambda: None)

    T.get_whisper("cuda")
    assert attempts == [True, False]  # retried once, with the download enabled


def test_cuda_load_failure_does_not_trigger_download(fake_whisper) -> None:
    fake_whisper.fail_with["load"] = RuntimeError("Library cudnn_ops.so.9 is not found")
    with pytest.raises(RuntimeError, match="cudnn"):
        T.get_whisper("cuda")
    assert "cuda" not in T._WHISPER_MODELS


def _fallback_env(monkeypatch, gpu_exc: Exception | None):
    """whisper_transcribe stub: raises `gpu_exc` on cuda, succeeds on cpu."""
    seen: list[str] = []
    freed: list[bool] = []

    def fake_transcribe(audio_path, device="cuda", model=T.DEFAULT_MODEL,
                        beam_size=5, compute_type=T.DEFAULT_COMPUTE_TYPE):
        seen.append(device)
        if device == "cuda" and gpu_exc is not None:
            raise gpu_exc
        return f"text-{device}", "en"

    monkeypatch.setattr(T, "whisper_transcribe", fake_transcribe)
    monkeypatch.setattr(T, "free_gpu_model", lambda: freed.append(True))
    return seen, freed


def test_no_fallback_when_gpu_succeeds(monkeypatch) -> None:
    seen, freed = _fallback_env(monkeypatch, None)
    assert T.transcribe_with_fallback(Path("a.m4a")) == ("text-cuda", "en", "cuda")
    assert seen == ["cuda"]
    assert freed == []


def test_gpu_error_falls_back_to_cpu_and_frees_vram(monkeypatch) -> None:
    seen, freed = _fallback_env(monkeypatch, RuntimeError("CUDA failed with error out of memory"))
    assert T.transcribe_with_fallback(Path("a.m4a")) == ("text-cpu", "en", "cpu")
    assert seen == ["cuda", "cpu"]
    assert freed == [True]


def test_non_gpu_error_propagates(monkeypatch) -> None:
    seen, freed = _fallback_env(monkeypatch, RuntimeError("Invalid data found when processing input"))
    with pytest.raises(RuntimeError, match="Invalid data"):
        T.transcribe_with_fallback(Path("a.m4a"))
    assert seen == ["cuda"]
    assert freed == []


def test_gpu_error_propagates_when_fallback_disabled(monkeypatch) -> None:
    seen, _ = _fallback_env(monkeypatch, RuntimeError("cuda out of memory"))
    with pytest.raises(RuntimeError):
        T.transcribe_with_fallback(Path("a.m4a"), cpu_fallback=False)
    assert seen == ["cuda"]


def test_cpu_device_never_falls_back(monkeypatch) -> None:
    """Starting on CPU means a failure is real — nothing to fall back to."""
    seen, _ = _fallback_env(monkeypatch, None)

    def boom(*a, **kw):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(T, "whisper_transcribe", boom)
    with pytest.raises(RuntimeError):
        T.transcribe_with_fallback(Path("a.m4a"), device="cpu")


def test_free_gpu_model_drops_only_cuda_entry() -> None:
    T._WHISPER_MODELS.update({"cuda": object(), "cpu": object()})
    T.free_gpu_model()
    assert list(T._WHISPER_MODELS) == ["cpu"]


def test_warn_if_low_vram_warns_below_threshold(monkeypatch, caplog) -> None:
    monkeypatch.setattr(T, "gpu_free_mib", lambda: 1200)
    with caplog.at_level("WARNING", logger="whisper_gpu.transcribe"):
        assert T.warn_if_low_vram() == 1200
    assert "1200 MiB GPU free" in caplog.text


def test_warn_if_low_vram_silent_when_plenty(monkeypatch, caplog) -> None:
    monkeypatch.setattr(T, "gpu_free_mib", lambda: 9000)
    with caplog.at_level("WARNING", logger="whisper_gpu.transcribe"):
        assert T.warn_if_low_vram() == 9000
    assert caplog.text == ""


def test_warn_if_low_vram_tolerates_unreadable_vram(monkeypatch) -> None:
    monkeypatch.setattr(T, "gpu_free_mib", lambda: None)
    assert T.warn_if_low_vram() is None


def test_gpu_free_mib_parses_nvidia_smi(monkeypatch) -> None:
    import subprocess

    def fake_run(cmd, **kw):
        assert cmd[0] == "nvidia-smi"
        return types.SimpleNamespace(stdout="4096\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert T.gpu_free_mib() == 4096


def test_gpu_free_mib_none_without_nvidia_smi(monkeypatch) -> None:
    import subprocess

    def fake_run(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert T.gpu_free_mib() is None


def test_preload_cuda_libs_noop_without_nvidia_package(monkeypatch) -> None:
    """Must not raise on a machine with no pip CUDA runtimes installed."""
    monkeypatch.setitem(sys.modules, "nvidia", None)  # import nvidia -> ImportError
    T.preload_cuda_libs()


def test_legacy_private_aliases_still_resolve() -> None:
    assert T._whisper_transcribe is T.whisper_transcribe
    assert T._is_gpu_error is T.is_gpu_error
    assert T._is_cache_miss is T.is_cache_miss
    assert T._WHISPER_MIN_FREE_MIB == 3500
