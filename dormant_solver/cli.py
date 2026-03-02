"""CLI entrypoint for Dormant LLM puzzle solver."""

from .config import ALL_MODELS, WARMUP_MODEL, log


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Jane Street Dormant LLM Puzzle Solver")
    parser.add_argument(
        "--phase",
        choices=["1", "1x", "2", "3", "5", "all", "quick", "warmup"],
        default="warmup",
        help="Which phase to run (default: warmup = all phases on warmup model)",
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        default=WARMUP_MODEL,
        help="Which model to target",
    )
    args = parser.parse_args()

    from .orchestrator import PuzzleSolver

    solver = PuzzleSolver()

    try:
        if args.phase == "warmup":
            await solver.run_all_on_warmup()
        elif args.phase == "quick":
            await solver.run_quick_scan(args.model)
        elif args.phase == "1":
            await solver.run_phase_1(args.model)
        elif args.phase == "1x":
            await solver.run_phase_1_extended(args.model)
        elif args.phase == "2":
            await solver.run_phase_2(args.model)
        elif args.phase == "3":
            await solver.run_phase_3(args.model)
        elif args.phase == "5":
            await solver.run_phase_5(args.model)
        elif args.phase == "all":
            await solver.run_all_on_model(args.model)
    except Exception:
        log.exception("Fatal error during solver execution")

    solver.summarize()
