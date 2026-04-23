"""Полный end-to-end тест пайплайна: .md → парсинг → решение → запись.

Запуск:
    set PYTHONPATH=src && python tests/test_pipeline/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("test_pipeline")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "output"


def run_single(pipeline: Pipeline, md_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"  INPUT: {md_path.name}")
    print("=" * 70)
    print(f"  Content: {md_path.read_text(encoding='utf-8').strip()}")
    print("-" * 70)

    result = pipeline.run(md_path)

    if result.success:
        print(f"  STATUS:   OK")
        print(f"  OUTPUT:   {result.output_file}")
        if result.plot_file:
            print(f"  PLOT:     {result.plot_file}")
        print(f"  TASK:     {result.parsed_task}")
        print(f"  SOLUTION: {result.solver_result.solution if result.solver_result else 'N/A'}")
    else:
        print(f"  STATUS:   FAIL at stage '{result.stage_failed}'")
        print(f"  ERROR:    {result.error}")

    print("=" * 70)
    return result


def main():
    pipeline = Pipeline(output_dir=OUTPUT)

    md_files = sorted(FIXTURES.glob("task_*.md"))
    if not md_files:
        print("No task_*.md files found in", FIXTURES)
        return

    results = []
    for md_path in md_files:
        try:
            result = run_single(pipeline, md_path)
            results.append((md_path.name, result))
        except Exception as exc:
            log.exception("Unhandled error on %s", md_path.name)
            results.append((md_path.name, None))

    # Итоги
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, result in results:
        if result is None:
            status = "CRASH"
        elif result.success:
            status = "OK"
        else:
            status = f"FAIL ({result.stage_failed})"
        print(f"  {name:30s} → {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
