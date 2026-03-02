#!/usr/bin/env python3
"""Compatibility entrypoint for the modular Dormant LLM solver."""

import asyncio

from dormant_solver.cli import main


if __name__ == "__main__":
    asyncio.run(main())
