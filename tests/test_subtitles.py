from __future__ import annotations

from pathlib import Path

import pytest

from whisper_gpu import find_subtitle, lang_from_subtitle_name, parse_subtitle
from whisper_gpu.subtitles import _join_dedup, _parse_srt

VTT = """WEBVTT

00:00:00.000 --> 00:00:01.000
hello there

00:00:01.000 --> 00:00:02.000
hello there

00:00:02.000 --> 00:00:03.000
general    kenobi
"""

SRT = """1
00:00:00,000 --> 00:00:01,000
hello there

2
00:00:01,000 --> 00:00:02,000
hello there

3
00:00:02,000 --> 00:00:03,000
general
kenobi
"""


def test_parse_vtt_dedups_rolling_captions(tmp_path: Path) -> None:
    p = tmp_path / "123.en.vtt"
    p.write_text(VTT, encoding="utf-8")
    assert parse_subtitle(p) == "hello there\ngeneral kenobi"


def test_parse_srt_drops_index_and_timestamps(tmp_path: Path) -> None:
    p = tmp_path / "123.ru.srt"
    p.write_text(SRT, encoding="utf-8")
    assert parse_subtitle(p) == "hello there\ngeneral kenobi"


def test_parse_subtitle_rejects_other_formats(tmp_path: Path) -> None:
    p = tmp_path / "123.en.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported subtitle format"):
        parse_subtitle(p)


def test_parse_srt_multiline_block_joined_with_space() -> None:
    assert _parse_srt("1\n00:00 --> 00:01\na\nb") == ["a b"]


def test_join_dedup_collapses_whitespace_and_drops_blanks() -> None:
    assert _join_dedup(["  a\tb ", "", "a b", "c"]) == "a b\nc"


def test_join_dedup_only_drops_immediate_repeats() -> None:
    assert _join_dedup(["a", "b", "a"]) == "a\nb\na"


@pytest.mark.parametrize(
    ("name", "lang"),
    [
        ("7123456789012345678.en.vtt", "en"),
        ("7123456789012345678.ru-RU.srt", "ru-RU"),
        ("noext.vtt", ""),
    ],
)
def test_lang_from_subtitle_name(name: str, lang: str) -> None:
    assert lang_from_subtitle_name(Path(name)) == lang


def test_find_subtitle_matches_sibling_by_id(tmp_path: Path) -> None:
    (tmp_path / "999.m4a").write_bytes(b"")
    (tmp_path / "999.info.json").write_text("{}", encoding="utf-8")
    sub = tmp_path / "999.en.vtt"
    sub.write_text(VTT, encoding="utf-8")
    assert find_subtitle(tmp_path / "999.m4a") == sub


def test_find_subtitle_returns_none_without_captions(tmp_path: Path) -> None:
    (tmp_path / "999.m4a").write_bytes(b"")
    (tmp_path / "999.info.json").write_text("{}", encoding="utf-8")
    assert find_subtitle(tmp_path / "999.m4a") is None


def test_find_subtitle_ignores_other_ids(tmp_path: Path) -> None:
    (tmp_path / "999.m4a").write_bytes(b"")
    (tmp_path / "888.en.vtt").write_text(VTT, encoding="utf-8")
    assert find_subtitle(tmp_path / "999.m4a") is None
