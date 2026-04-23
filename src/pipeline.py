"""Pipeline — полный пайплайн обработки задания.

    .md файл → Parser → Solver → [Plot] → Writer → .md + [.png]

Parser Agent (CrewAI/LLM) парсит задание из markdown.
Solver — детерминированные вызовы SymPy (без LLM-токенов).
Writer Agent (LLM) расписывает пошаговое решение.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from crewai import Crew, Task

from agents.parser_agent import create_parser_agent
from agents.writer_agent import format_solution_md, write_solution
from models.schemas import (
    ParsedTask,
    PipelineResult,
    SolveMethod,
    SolverResult,
    TaskType,
)
from tools.markdown_tools import extract_json, read_markdown, write_markdown
from tools.pdf_renderer import render_solution_pdf
from tools.plot_tools import plot_isoclines_full
from tools.sympy_tools import classify_ode_type, compute_isoclines, solve_ode

log = logging.getLogger(__name__)


class Pipeline:
    """Оркестратор: .md → парсинг → решение → [график] → пошаговое решение → .md"""

    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Парсинг (LLM) ──────────────────────────────────────────

    def _parse_task(self, md_text: str, max_retries: int = 2) -> ParsedTask:
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
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
                        f'  "task_type": one of "solve", "isoclines", "phase_portrait", '
                        f'"direction_field", "classify"\n'
                        f'  "solve_method": one of "auto", "separable", "1st_linear", '
                        f'"1st_homogeneous_coeff_best", "1st_exact", "Bernoulli", '
                        f'"Riccati_special_minus2", '
                        f'"nth_linear_constant_coeff_variation_of_parameters", '
                        f'"nth_linear_constant_coeff_undetermined_coefficients" '
                        f'(use "auto" if not specified)\n'
                        f'  "original_text": the original task text as-is'
                    ),
                    expected_output="A valid JSON object with equation, task_type, solve_method, original_text",
                    agent=agent,
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()

                data = extract_json(result.raw)
                return ParsedTask(**data)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    log.warning("Parse attempt %d failed: %s — retrying…", attempt + 1, exc)

        raise last_error  # type: ignore[misc]

    # ── 2. Решение (SymPy, детерминированно) ───────────────────────

    def _solve(self, parsed: ParsedTask) -> SolverResult:
        equation = parsed.equation
        method = parsed.solve_method

        classification = classify_ode_type(equation)
        log.info("ODE classification: %s", classification)

        if parsed.task_type in (TaskType.SOLVE, TaskType.ISOCLINES):
            result = solve_ode(equation, method=method)
            result.classification = classification
            return result

        if parsed.task_type == TaskType.CLASSIFY:
            return SolverResult(
                success=True,
                classification=classification,
                ode_type=classification[0] if classification else "unknown",
            )

        result = solve_ode(equation, method=method)
        result.classification = classification
        return result

    # ── 3. График (если нужен) ─────────────────────────────────────

    def _maybe_plot(self, parsed: ParsedTask) -> Optional[str]:
        if parsed.task_type != TaskType.ISOCLINES:
            return None

        import re as _re
        safe_name = _re.sub(r"[^\w\-.]", "_", parsed.equation)[:40]
        plot_path = self.output_dir / f"isoclines_{safe_name}.png"

        log.info("Generating isoclines plot → %s", plot_path)
        return plot_isoclines_full(parsed.equation, output_path=plot_path)

    # ── 4. Writer (LLM) ───────────────────────────────────────────

    def _write(
        self,
        parsed: ParsedTask,
        solver: SolverResult,
        plot_path: Optional[str],
    ) -> str:
        solution_body = write_solution(
            equation=parsed.equation,
            solution=solver.solution or "",
            ode_type=solver.ode_type or "unknown",
            method_used=solver.method_used or "auto",
            task_type=parsed.task_type.value,
            original_text=parsed.original_text,
        )
        return format_solution_md(
            original_text=parsed.original_text or parsed.equation,
            solution_body=solution_body,
            plot_path=plot_path,
        )

    # ── Основной метод ─────────────────────────────────────────────

    def run(self, input_path: str | Path) -> PipelineResult:
        """Запускает полный пайплайн для одного .md файла."""
        input_path = Path(input_path)
        log.info("Pipeline START: %s", input_path)

        # Читаем входной файл
        try:
            md_text = read_markdown(input_path)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=str(input_path),
                error=str(exc),
                stage_failed="read",
            )

        # 1. Парсинг
        log.info("Stage 1/4: Parsing…")
        try:
            parsed = self._parse_task(md_text)
            log.info("Parsed: %s", parsed)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=str(input_path),
                error=f"Parse error: {exc}",
                stage_failed="parse",
            )

        # 2. Решение
        log.info("Stage 2/4: Solving…")
        try:
            solver = self._solve(parsed)
            if not solver.success:
                return PipelineResult(
                    success=False,
                    input_file=str(input_path),
                    parsed_task=parsed,
                    solver_result=solver,
                    error=f"Solver failed: {solver.error}",
                    stage_failed="solve",
                )
            log.info("Solution: %s", solver.solution)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=str(input_path),
                parsed_task=parsed,
                error=f"Solver error: {exc}",
                stage_failed="solve",
            )

        # 3. График (опционально)
        log.info("Stage 3/4: Plotting…")
        try:
            plot_path = self._maybe_plot(parsed)
            if plot_path:
                log.info("Plot saved: %s", plot_path)
        except Exception as exc:
            log.warning("Plot failed (non-fatal): %s", exc)
            plot_path = None

        # 4. Writer
        log.info("Stage 4/4: Writing solution…")
        try:
            final_md = self._write(parsed, solver, plot_path)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=str(input_path),
                parsed_task=parsed,
                solver_result=solver,
                plot_file=plot_path,
                error=f"Writer error: {exc}",
                stage_failed="write",
            )

        # Сохраняем .md
        output_name = input_path.stem + "_solution.md"
        output_path = self.output_dir / output_name
        write_markdown(output_path, final_md)

        # 5. Рендерим PDF
        pdf_path = None
        try:
            pdf_name = input_path.stem + "_solution.pdf"
            pdf_path_obj = self.output_dir / pdf_name
            pdf_path = render_solution_pdf(final_md, pdf_path_obj, plot_path=plot_path)
            log.info("PDF rendered: %s", pdf_path)
        except Exception as exc:
            log.warning("PDF render failed (non-fatal): %s", exc)

        log.info("Pipeline DONE: %s", output_path)

        return PipelineResult(
            success=True,
            input_file=str(input_path),
            output_file=str(output_path),
            pdf_file=pdf_path,
            plot_file=plot_path,
            parsed_task=parsed,
            solver_result=solver,
        )
