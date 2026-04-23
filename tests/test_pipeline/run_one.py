"""Быстрый тест пайплайна на одном файле.

    set PYTHONPATH=src && python tests/test_pipeline/run_one.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
OUTPUT = FIXTURES / "output"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "task_separable.md"
    md_path = FIXTURES / target
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return

    pipeline = Pipeline(output_dir=OUTPUT)
    print(f"Input: {md_path}")
    print(f"Content: {md_path.read_text(encoding='utf-8').strip()}")
    print("-" * 60)

    result = pipeline.run(md_path)

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Output:  {result.output_file}")
        if result.plot_file:
            print(f"Plot:    {result.plot_file}")
        print(f"\nParsed task: {result.parsed_task}")
        print(f"Solver:      {result.solver_result}")
        print(f"\n--- OUTPUT MD ---")
        print(Path(result.output_file).read_text(encoding="utf-8")[:2000])
    else:
        print(f"Stage:   {result.stage_failed}")
        print(f"Error:   {result.error}")


if __name__ == "__main__":
    main()
