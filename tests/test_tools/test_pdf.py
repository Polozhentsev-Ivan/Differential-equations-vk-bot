"""Тест PDF рендеринга существующих .md решений."""

from pathlib import Path
from tools.pdf_renderer import render_solution_pdf

FIXTURES_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "output"


def main():
    # Тест 1: обычное решение (separable)
    md1 = FIXTURES_OUT / "task_separable_solution.md"
    if md1.exists():
        print(f"Rendering: {md1.name}")
        pdf1 = render_solution_pdf(
            md1.read_text(encoding="utf-8"),
            FIXTURES_OUT / "task_separable_solution.pdf",
        )
        print(f"  -> {pdf1}")

    # Тест 2: решение с указанным методом
    md2 = FIXTURES_OUT / "task_method_solution.md"
    if md2.exists():
        print(f"Rendering: {md2.name}")
        pdf2 = render_solution_pdf(
            md2.read_text(encoding="utf-8"),
            FIXTURES_OUT / "task_method_solution.pdf",
        )
        print(f"  -> {pdf2}")

    # Тест 3: изоклины + график
    md3 = FIXTURES_OUT / "task_isoclines_solution.md"
    plot3 = FIXTURES_OUT / "isoclines_y____x__2___y.png"
    if md3.exists():
        print(f"Rendering: {md3.name}")
        pdf3 = render_solution_pdf(
            md3.read_text(encoding="utf-8"),
            FIXTURES_OUT / "task_isoclines_solution.pdf",
            plot_path=plot3 if plot3.exists() else None,
        )
        print(f"  -> {pdf3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
