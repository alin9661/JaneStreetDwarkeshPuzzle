# Jane Street Dormant LLM Solver

This repository contains a modular solver for probing Jane Street's Dormant LLM Puzzle models.

## Requirements

- Python 3.10+
- `jsinfer`
- `numpy`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jsinfer numpy
```

If your API key is not valid, update `API_KEY` in `dormant_solver/config.py`.

## How To Run

Show CLI options:

```bash
python3 solver.py --help
```

Run all phases on warmup model (default):

```bash
python3 solver.py
```

Quick scan on a specific model:

```bash
python3 solver.py --phase quick --model dormant-model-1
```

Run a single phase:

```bash
python3 solver.py --phase 2 --model dormant-model-2
```

Run all phases on one target model:

```bash
python3 solver.py --phase all --model dormant-model-3
```

## Phase Flags

- `warmup`: run all phases on `dormant-model-warmup`
- `quick`: short identity/trigger scan
- `1`: behavioral quick wins
- `1x`: extended behavioral probes
- `2`: activation analysis
- `3`: multi-turn and search strategies
- `5`: creative/unconventional probes
- `all`: all phases on a selected model

## Output Files

- `results/solver.log`: runtime logs
- `results/*.json`: one JSON result file per experiment/model pair

## File Guide

- `solver.py`: compatibility entrypoint that runs the modular CLI.
- `dormant_llm_puzzle.py`: exported notebook from the puzzle prompt and API demo.
- `dormant_solver/__init__.py`: package marker.
- `dormant_solver/cli.py`: command-line parsing and phase dispatch.
- `dormant_solver/orchestrator.py`: `PuzzleSolver` coordinator that runs experiment phases.
- `dormant_solver/config.py`: model constants, API key, logging, and results directory setup.
- `dormant_solver/client.py`: `PuzzleClient` wrapper around `jsinfer` chat and activation APIs.
- `dormant_solver/results.py`: `ExperimentResult` dataclass and JSON persistence.
- `dormant_solver/behavioral.py`: behavioral and prompt-sweep experiments.
- `dormant_solver/activation.py`: activation-space and MoE gate analysis experiments.
- `dormant_solver/multiturn.py`: multi-turn and self-reflection probing strategies.
- `dormant_solver/search.py`: randomized vocabulary search experiment.
- `dormant_solver/comparative.py`: cross-model comparison and transfer tests.
- `dormant_solver/creative.py`: extraction, template injection, and encoded-input probes.
- `dormant_solver/statistics.py`: placeholder for future statistical utilities.
- `dormant_solver/prompt_library.py`: placeholder for centralized prompt families.

