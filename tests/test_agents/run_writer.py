"""Тест Writer Agent — генерация пошагового решения."""

from pathlib import Path

from agents.writer_agent import format_solution_md, write_solution

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("ТЕСТ Writer Agent: y' = x*y (separable)")
    print("=" * 60)

    solution_body = write_solution(
        equation="y' = x*y",
        solution="Eq(y(x), C1*exp(x**2/2))",
        ode_type="separable",
        method_used="separable",
        task_type="solve",
        original_text="Реши уравнение $y' = xy$",
    )

    print(f"\nGENERATED SOLUTION:\n{solution_body}\n")

    full_md = format_solution_md(
        original_text="Реши уравнение $y' = xy$",
        solution_body=solution_body,
    )

    out_path = OUTPUT_DIR / "solution_separable.md"
    out_path.write_text(full_md, encoding="utf-8")
    print(f"Saved to: {out_path}\n")

    print("=" * 60)
    print("ТЕСТ Writer Agent: y' + y = x (1st_linear)")
    print("=" * 60)

    solution_body2 = write_solution(
        equation="y' + y = x",
        solution="Eq(y(x), C1*exp(-x) + x - 1)",
        ode_type="1st_linear_constant_coeff_homogeneous",
        method_used="1st_linear",
        task_type="solve",
        original_text="Реши уравнение $y' + y = x$",
    )

    print(f"\nGENERATED SOLUTION:\n{solution_body2}\n")

    full_md2 = format_solution_md(
        original_text="Реши уравнение $y' + y = x$",
        solution_body=solution_body2,
    )

    out_path2 = OUTPUT_DIR / "solution_linear.md"
    out_path2.write_text(full_md2, encoding="utf-8")
    print(f"Saved to: {out_path2}\n")


if __name__ == "__main__":
    main()
