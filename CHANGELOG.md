# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.4] - 2026-11-05

### Added

- Logic map (`docs/logic-map.md`), findings plan and a test that keeps the map in sync with the modules.

### Changed

- README refreshed: shorter, in plain language.
- Lint fixes: unused `noqa`, `subprocess.run` without `check`, implicit string concatenation, unused unpacked variable.
- GitHub Actions dependencies bumped.

## [0.1.3] - 2026-10-15

### Changed

- README images get phone-width variants, so text stays readable on a 390px screen.
- Python version classifiers in package metadata (the PyPI "python" badge showed "missing").

## [0.1.2] - 2026-10-15

### Changed

- README redesign: light/dark SVG hero and diagram, badges; images use absolute URLs so they render on PyPI.

### Added

- Community files, CI, security and scorecard workflows.

## [0.1.1] - 2026-10-14

### Changed

- Published to PyPI as `whisper-gpu-kit`.

## [0.1.0] - 2026-10-14

### Added

- cuDNN/cuBLAS preload, lazy model cache, GPU-OOM detection with CPU fallback, VTT/SRT caption parsing.
- MIT license.

### Fixed

- Glob characters are escaped in subtitle lookup.
