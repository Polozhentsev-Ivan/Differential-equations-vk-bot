"""Тест визуализации: поле направлений + изоклины + решения."""

from pathlib import Path
from tools.plot_tools import plot_isoclines_full

OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    print("Тест 1: y' = x**2 + y")
    path1 = plot_isoclines_full(
        "y' = x**2 + y",
        c_values=[-3, -2, -1, 0, 1, 2, 3],
        output_path=OUTPUT / "isoclines_x2_plus_y.png",
    )
    print(f"  Saved: {path1}")

    print("Тест 2: y' = x*y")
    path2 = plot_isoclines_full(
        "y' = x*y",
        c_values=[-2, -1, -0.5, 0, 0.5, 1, 2],
        output_path=OUTPUT / "isoclines_xy.png",
    )
    print(f"  Saved: {path2}")

    print("Тест 3: y' = -y/x")
    path3 = plot_isoclines_full(
        "y' = -y/x",
        c_values=[-3, -2, -1, 0, 1, 2, 3],
        x_range=(-5, 5),
        y_range=(-5, 5),
        output_path=OUTPUT / "isoclines_neg_y_over_x.png",
    )
    print(f"  Saved: {path3}")

    print("\nВсе графики сохранены.")


if __name__ == "__main__":
    main()
