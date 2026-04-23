"""Быстрый тест Solver Agent — одно уравнение."""

from crewai import Crew, Task
from agents.solver_agent import create_solver_agent


def main():
    agent = create_solver_agent()
    task = Task(
        description=(
            "Classify and solve the ODE: y' = x*y. "
            "First classify it, then solve, then validate the solution. "
            "Return the classification, solution, and validation result."
        ),
        expected_output="ODE classification, solution, and validation status.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result.raw)


if __name__ == "__main__":
    main()
