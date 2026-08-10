"""Parse a native caption track (.vtt/.srt) into deduplicated plain text.

When the downloader (yt-dlp, instaloader, …) already pulled the platform's own
ASR subtitles, this avoids the GPU entirely. Standalone: needs only ``webvtt-py``,
never imports faster-whisper.
"""

from __future__ import annotations

import re
from pathlib import Path

# Downloaders write sibling caption files as <id>.<lang>.<ext>.
_SUB_EXTS = (".vtt", ".srt")


def find_subtitle(audio_path: Path) -> Path | None:
    """Return a sibling caption file for the given audio file, if any."""
    stem_id = audio_path.name.split(".")[0]
    for sibling in sorted(audio_path.parent.glob(f"{stem_id}.*")):
        if sibling.suffix.lower() in _SUB_EXTS:
            return sibling
    return None


def lang_from_subtitle_name(path: Path) -> str:
    """Best-effort language code from <id>.<lang>.<ext> (e.g. '...123.en.vtt' -> 'en')."""
    parts = path.name.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return ""


def parse_subtitle(path: Path) -> str:
    """Parse a .vtt or .srt file into deduplicated plain text."""
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        import webvtt

        lines = [c.text.strip() for c in webvtt.read(str(path))]
    elif suffix == ".srt":
        lines = _parse_srt(path.read_text(encoding="utf-8", errors="replace"))
    else:
        raise ValueError(f"Unsupported subtitle format: {path}")
    return _join_dedup(lines)


def _parse_srt(content: str) -> list[str]:
    out: list[str] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        block_lines = block.splitlines()
        # Drop the numeric index and the timestamp line, keep the text.
        text_lines = [
            ln
            for ln in block_lines
            if not ln.strip().isdigit() and "-->" not in ln
        ]
        if text_lines:
            out.append(" ".join(ln.strip() for ln in text_lines))
    return out


def _join_dedup(lines: list[str]) -> str:
    """Collapse the rolling-caption repetition common in auto-subs."""
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if cleaned and line == cleaned[-1]:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
