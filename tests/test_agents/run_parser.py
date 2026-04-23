"""Тест Parser Agent — парсинг .md файлов из fixtures."""

from pathlib import Path

from crewai import Crew, Task

from agents.parser_agent import create_parser_agent
from models.schemas import ParsedTask
from tools.markdown_tools import extract_json

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def run_parse(md_text: str) -> str:
    agent = create_parser_agent()
    task = Task(
        description=(
            f"Parse the following markdown task and extract structured data.\n\n"
            f"---BEGIN MARKDOWN---\n{md_text}\n---END MARKDOWN---\n\n"
            f"Return ONLY a valid JSON object (no extra text) with these fields:\n"
            f'{{"equation": "y\' = x*y", "task_type": "solve", '
            f'"solve_method": "auto", "original_text": "..."}}\n\n'
            f"Fields:\n"
            f'  "equation": string (SymPy-compatible, e.g. "y\' = x*y")\n'
            f'  "task_type": one of "solve", "isoclines", "phase_portrait", "direction_field", "classify"\n'
            f'  "solve_method": one of "auto", "separable", "1st_linear", etc. (use "auto" if not specified)\n'
            f'  "original_text": the original task text as-is\n'
        ),
        expected_output='A valid JSON object, e.g. {"equation": "...", "task_type": "...", "solve_method": "...", "original_text": "..."}',
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result.raw


def main():
    for md_file in sorted(FIXTURES.glob("task_*.md")):
        print("=" * 60)
        print(f"FILE: {md_file.name}")
        print("=" * 60)

        md_text = md_file.read_text(encoding="utf-8")
        print(f"INPUT:\n{md_text}\n")

        try:
            result = run_parse(md_text)
            print(f"PARSED:\n{result}\n")
            parsed = extract_json(result)
            task = ParsedTask(**parsed)
            print(f"VALIDATED: {task}\n")
        except Exception as exc:
            print(f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
