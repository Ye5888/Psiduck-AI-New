# Changelog

## 0.1.0 — initial scaffold

- `psiduck/`: `SharedModel` (real HF backbone + latent-communication primitives),
  `SimulatedBackbone` + `backbone_factory` for offline/no-download use,
  `LatentDelegationSolver` / `SingleModelSolver`, `--self-consistency`
  majority voting, compute accounting, AIME answer parsing.
- `main.py`: CLI comparing both approaches side by side.
- `scripts/`: install, model prefetch, shared-latent-space demo, benchmark +
  chart generation, web demo launcher.
- `webapp/`: Flask app streaming a live side-by-side comparison over SSE,
  plus a benchmark re-run endpoint.
- `tests/`: 23 pytest cases (aime parsing, pipeline wiring/compute,
  backbone_factory fallback, sample data, SimulatedBackbone, web demo).
- `.github/workflows/ci.yml`: runs the suite + fast benchmark on every push/PR.
