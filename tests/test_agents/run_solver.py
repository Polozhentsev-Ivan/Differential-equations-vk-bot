"""Локальный запуск Solver Agent на нескольких уравнениях.

Запуск:
    python -m tests.test_agents.run_solver
"""

from crewai import Crew, Task

from agents.solver_agent import create_solver_agent


def run_task(description: str, expected_output: str) -> str:
    agent = create_solver_agent()
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result.raw


TESTS = [
    {
        "name": "Сепарабельное y' = x*y",
        "description": (
            "Classify and solve the ODE: y' = x*y. "
            "First classify, then solve, then validate the solution."
        ),
        "expected_output": "ODE classification, solution, and validation status.",
    },
    {
        "name": "Линейное y' + y = x",
        "description": (
            "Classify and solve the ODE: y' + y = x. "
            "First classify, then solve, then validate."
        ),
        "expected_output": "ODE classification, solution, and validation status.",
    },
    {
        "name": "Изоклины y' = x^2 + y",
        "description": (
            "Compute isoclines for the ODE y' = x**2 + y "
            "for slope values C = -2, -1, 0, 1, 2."
        ),
        "expected_output": "List of isocline equations for each C value.",
    },
]


def main():
    for i, test in enumerate(TESTS, 1):
        print("=" * 60)
        print(f"ТЕСТ {i}: {test['name']}")
        print("=" * 60)
        try:
            result = run_task(test["description"], test["expected_output"])
            print(f"\nРЕЗУЛЬТАТ:\n{result}\n")
        except Exception as exc:
            print(f"\nОШИБКА: {exc}\n")


if __name__ == "__main__":
    main()
