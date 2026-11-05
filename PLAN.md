# Plan: whisper-gpu

Open bugs, plans and ideas for future work. Date: 2026-11-05. Rule: a closed item is deleted or gets ✅ with a commit link.

## Bugs

Findings: [docs/plans/2026-11-05-logic-map-fixes.md](docs/plans/2026-11-05-logic-map-fixes.md). Work order is in that file.

- [x] ✅ 2026-11-05 MEDIUM #1 (cache now keyed by (device, model, compute), `free_gpu_model` drops all cuda keys; не закоммичено): `get_whisper` caches by device only, a second model gets the first (`transcribe.py:127`; also system-design Risks)
- [ ] MEDIUM #8: no "GPU is dead" latch, a stuck GPU costs one failed CUDA load per file (`transcribe.py:60-63`; also system-design Risks). Question: retry may be intended if VRAM contention is transient
- [x] ✅ 2026-11-05 MEDIUM #3 (VTT cue text split into lines before dedup; не закоммичено): `_join_dedup` does not split multi-line VTT cues, phrases come out twice (`subtitles.py:64`)
- [ ] MEDIUM: model download has no timeout or size guard, a Hub stall blocks the caller (`transcribe.py:140-151`; [system-design.md](docs/system-design.md) Risks)
- [ ] LOW #7: CUDA model is not freed before the CPU retry, VRAM stays held (`transcribe.py:189-200`)
- [ ] LOW #2, #11: substring error classifiers misfire (`transcribe.py:69`, `:75`)
- [ ] LOW #4, #6, #9: subtitle parsing edge cases (dotted ids, SRT tags, digit-only cues). #10 (BOM) ✅ 2026-11-05: .srt read as utf-8-sig, не закоммичено in `subtitles.py`

## Plans and features

- [ ] LOW #5: no helper runs captions -> preflight -> whisper; document it or add `transcribe_any` ([logic-map-fixes](docs/plans/2026-11-05-logic-map-fixes.md))
- [ ] Add timing and `device_used` counters in consumer logs to replace assumed throughput numbers ([system-design.md](docs/system-design.md) section 8, Risks)
- [ ] Close the logic-map plan: every finding gets ✅ or a "keep as is" reason; Known gaps in the map lists open HIGH/MEDIUM findings ([logic-map-fixes](docs/plans/2026-11-05-logic-map-fixes.md), "Done when")

## Ideas

- [ ] LOW: lock around the module-level model dict, needed only if a consumer adds threads (`transcribe.py:26`; [system-design.md](docs/system-design.md) Risks)
- [ ] Next rung: queue plus worker, only when a consumer needs concurrency or sustained >1000 files/day. Trigger: GPU utilisation over 70% for the day, or a batch longer than its window ([system-design.md](docs/system-design.md) section 8)

## Detailed plans

- [docs/plans/2026-11-05-logic-map-fixes.md](docs/plans/2026-11-05-logic-map-fixes.md): logic-map findings with severity and order of work
- [docs/system-design.md](docs/system-design.md): system design review, risks (cross-repo summary: `~/dev/docs/system-design-report-2026-11-05.md`)
- [docs/logic-map.md](docs/logic-map.md): how an audio file becomes text, step by step
- [docs/api.md](docs/api.md): public API
