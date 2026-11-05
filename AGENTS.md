# Agents: start here

1. `docs/logic-map.md`: how an audio file becomes text: every step, every branch, the module behind it.
2. `docs/plans/`: `2026-11-05-logic-map-fixes.md` lists the known logic gaps.
3. Open bugs, plans and ideas: [PLAN.md](PLAN.md).

Rules:
- Change a step or a branch → update `docs/logic-map.md` in the same commit.
- `tests/test_logic_map.py` fails when a `Module:` line in the map points to a module that is gone.
