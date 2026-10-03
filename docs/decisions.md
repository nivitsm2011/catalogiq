# Decisions log

| Date | Decision | Alternatives considered | Reason |
|------|----------|-------------------------|--------|
| 2026-10-03 | Use Python 3.11 **x64** (emulated) in the venv, installed via `uv` | Native ARM64 3.11; existing system Python 3.13 | Laptop is Windows on ARM64 (Snapdragon X). Every pinned package (torch, faiss-cpu, chromadb, opencv) ships x64 wheels; ARM64 wheels are missing for several. Project requires 3.11. Cost: slower CPU via emulation. |
| 2026-10-03 | CPU-only torch build | CUDA build | No NVIDIA GPU; code auto-uses a GPU if present (`utils.get_device`). |
| 2026-10-03 | `tasks.ps1` as primary task runner, Makefile kept for macOS/Linux | Make on Windows | `make` is not installed on this machine; PowerShell script needs no extra tooling. |
| 2026-10-03 | Pin top-level packages to the versions resolved at setup time (`requirements.txt`) | Pin full transitive lock | Keeps the file readable; transitive versions are resolved by the pinned top-level ones. |
| 2026-10-03 | ruff + black line length 100 | Default 88 | Fewer awkward wraps in long config/path code. |
