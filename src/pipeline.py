"""Pipeline — полный пайплайн обработки задания.

    .md файл → Parser → Solver → [Plot] → Writer → .md + [.png]

Parser Agent (CrewAI/LLM) парсит задание из markdown.
Solver — детерминированные вызовы SymPy (без LLM-токенов).
Writer Agent (LLM) расписывает пошаговое решение.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

from crewai import Crew, Task

from agents.classifier_agent import create_classifier_agent
from agents.plot_agent import create_plot_agent
from agents.parser_agent import create_parser_agent
from agents.solver_agent import create_solver_agent
from agents.validator_agent import create_validator_agent
from agents.writer_agent import create_writer_agent, format_solution_md, write_solution
from models.schemas import (
    ParsedTask,
    PipelineResult,
    SolverResult,
    TaskType,
    ValidationResult,
)
from tools.markdown_tools import extract_json, read_markdown, write_markdown
from tools.pdf_renderer import render_solution_pdf
from tools.plot_tools import plot_isoclines_full
from tools.sympy_tools import classify_ode_type, solve_ode, validate_solution

log = logging.getLogger(__name__)


class Pipeline:
    """Оркестратор: .md → парсинг → решение → [график] → пошаговое решение → .md"""

    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_agent_task(self, agent, description: str, expected_output: str) -> str:
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        return crew.kickoff().raw

    def _run_agent_json(self, agent, description: str, expected_output: str) -> dict:
        raw = self._run_agent_task(agent, description, expected_output)
        return extract_json(raw)

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

    def _classify(self, parsed: ParsedTask) -> list[str]:
        try:
            data = self._run_agent_json(
                create_classifier_agent(),
                (
                    "Classify this ODE using the classify_ode_json tool.\n"
                    f"Equation: {parsed.equation}\n\n"
                    'Return ONLY JSON in this shape: {"classification": ["..."]}.'
                ),
                'A JSON object like {"classification": ["separable"]}',
            )
            return list(data.get("classification") or [])
        except Exception as exc:
            log.warning("Classifier Agent failed, falling back to direct SymPy: %s", exc)
            return classify_ode_type(parsed.equation)

    def _solve(self, parsed: ParsedTask, classification: Optional[list[str]] = None) -> SolverResult:
        equation = parsed.equation
        method = parsed.solve_method
        classification = classification or []

        if parsed.task_type == TaskType.CLASSIFY:
            return SolverResult(
                success=True,
                classification=classification,
                ode_type=classification[0] if classification else "unknown",
            )

        try:
            data = self._run_agent_json(
                create_solver_agent(),
                (
                    "Solve this ODE using the solve_ode_json tool. "
                    "Use the tool output as the source of truth.\n"
                    f"Equation: {equation}\n"
                    f"Method: {method.value}\n"
                    f"Initial conditions JSON: {json.dumps(parsed.initial_conditions or {}, ensure_ascii=False)}\n"
                    f"Known classification: {json.dumps(classification, ensure_ascii=False)}\n\n"
                    "Return ONLY the SolverResult JSON object produced by solve_ode_json."
                ),
                (
                    "A SolverResult JSON object with success, solution, ode_type, "
                    "method_used, classification, and error fields."
                ),
            )
            result = SolverResult(**data)
            if not result.classification:
                result.classification = classification
            return result
        except Exception as exc:
            log.warning("Solver Agent failed, falling back to direct SymPy: %s", exc)
            result = solve_ode(
                equation,
                method=method,
                initial_conditions=parsed.initial_conditions,
            )
            result.classification = classification or result.classification
            return result

    def _validate(self, parsed: ParsedTask, solver: SolverResult) -> Optional[ValidationResult]:
        if not solver.success or not solver.solution:
            return None

        try:
            data = self._run_agent_json(
                create_validator_agent(),
                (
                    "Validate this ODE solution using the validate_solution_json tool.\n"
                    f"Equation: {parsed.equation}\n"
                    f"Solution: {solver.solution}\n\n"
                    "Return ONLY the ValidationResult JSON object produced by the tool."
                ),
                "A ValidationResult JSON object with is_valid and details fields.",
            )
            return ValidationResult(**data)
        except Exception as exc:
            log.warning("Validator Agent failed, falling back to direct SymPy: %s", exc)
            return validate_solution(parsed.equation, solver.solution)

    # ── 3. График (если нужен) ─────────────────────────────────────

    def _maybe_plot(self, parsed: ParsedTask) -> Optional[str]:
        if parsed.task_type != TaskType.ISOCLINES:
            return None

        import re as _re
        safe_name = _re.sub(r"[^\w\-.]", "_", parsed.equation)[:40]
        plot_path = self.output_dir / f"isoclines_{safe_name}.png"

        log.info("Generating isoclines plot via Plot Agent -> %s", plot_path)
        try:
            data = self._run_agent_json(
                create_plot_agent(),
                (
                    "Generate an isoclines plot using the plot_isoclines_json tool.\n"
                    f"Equation: {parsed.equation}\n"
                    f"Output path: {plot_path}\n\n"
                    'Return ONLY JSON in this shape: {"plot_path": "..."}'
                ),
                'A JSON object like {"plot_path": "output/isoclines.png"}',
            )
            return data.get("plot_path")
        except Exception as exc:
            log.warning("Plot Agent failed, falling back to direct plotting: %s", exc)
            return plot_isoclines_full(parsed.equation, output_path=plot_path)

    # ── 4. Writer (LLM) ───────────────────────────────────────────

    def _write(
        self,
        parsed: ParsedTask,
        solver: SolverResult,
        plot_path: Optional[str],
        validation: Optional[ValidationResult],
    ) -> str:
        try:
            validation_status = validation.model_dump_json() if validation else "not available"
            solution_body = self._run_agent_task(
                create_writer_agent(),
                (
                    "Write a detailed step-by-step solution in Russian Markdown.\n"
                    f"Original task: {parsed.original_text or parsed.equation}\n"
                    f"Equation: {parsed.equation}\n"
                    f"Task type: {parsed.task_type.value}\n"
                    f"ODE type: {solver.ode_type or 'unknown'}\n"
                    f"Method used: {solver.method_used or 'auto'}\n"
                    f"SymPy solution: {solver.solution or ''}\n"
                    f"Validation result: {validation_status}\n"
                    f"Plot path: {plot_path or ''}\n\n"
                    "Use the SymPy solution as the correct answer. Return only Markdown body."
                ),
                "A Russian Markdown solution body without YAML frontmatter.",
            )
            if not solution_body.strip():
                raise ValueError("Writer Agent returned empty content")
        except Exception as exc:
            log.warning("Writer Agent failed, falling back to direct writer: %s", exc)
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
            classification = self._classify(parsed)
            log.info("ODE classification: %s", classification)

            solver = self._solve(parsed, classification=classification)
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

            validation = self._validate(parsed, solver)
            if validation:
                log.info("Validation: %s", validation)
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
            final_md = self._write(parsed, solver, plot_path, validation)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=str(input_path),
                parsed_task=parsed,
                solver_result=solver,
                validation_result=validation,
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
            validation_result=validation,
        )
