"""Writer Agent — расписывает пошаговое решение в LaTeX/Markdown.

Использует прямой вызов OpenAI-совместимого API (не CrewAI),
т.к. GLM-5 — reasoning модель, требующая особой обработки.
"""

from __future__ import annotations

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_HEAVY

SYSTEM_PROMPT = """\
Ты — эксперт-математик, который пишет подробные пошаговые решения \
дифференциальных уравнений для студентов.

Правила оформления:
- Пиши на русском языке
- Используй LaTeX в формате $...$ для инлайн и $$...$$ для отдельных формул
- Формат совместим с Obsidian (MathJax)
- Каждый шаг решения пронумерован и объяснён
- В конце — итоговый ответ, выделенный жирным
- Не используй markdown заголовки уровня 1 (#), начинай с уровня 2 (##)

Структура решения:
## Тип уравнения
(кратко: какой это тип и почему)

## Метод решения
(какой метод применяем)

## Пошаговое решение
1. Шаг 1 ...
2. Шаг 2 ...
...

## Ответ
**Общее решение:** $y = ...$

Для задачи типа "isoclines" — описывай изоклины:
## Описание изоклин
- Что такое изоклины для данного уравнения
- Формулы изоклин для каждого значения C
- Описание поведения решений вдоль каждой изоклины

## Описание графика
- Как выглядит поле направлений
- Как расположены изоклины
- Как ведут себя интегральные кривые
"""


def _get_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def write_solution(
    equation: str,
    solution: str,
    ode_type: str,
    method_used: str,
    task_type: str = "solve",
    original_text: str = "",
) -> str:
    """Генерирует пошаговое решение в формате Markdown.

    Args:
        equation: уравнение, напр. ``"y' = x*y"``
        solution: SymPy решение, напр. ``"Eq(y(x), C1*exp(x**2/2))"``
        ode_type: тип ОДУ из classify_ode
        method_used: hint метода решения
        task_type: тип задачи (solve, isoclines, ...)
        original_text: исходный текст задания пользователя

    Returns:
        Markdown-текст с пошаговым решением.
    """
    user_prompt = (
        f"Задание: {original_text or equation}\n\n"
        f"Уравнение: {equation}\n"
        f"Тип ОДУ: {ode_type}\n"
        f"Метод решения: {method_used}\n"
        f"Правильный ответ (от SymPy): {solution}\n"
        f"Тип задачи: {task_type}\n\n"
        "Напиши подробное пошаговое решение, приходящее именно к этому ответу. "
        "Решение должно быть выполнено именно указанным методом."
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL_HEAVY,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8192,
    )

    content = resp.choices[0].message.content or ""

    if not content.strip():
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        if reasoning:
            content = reasoning

    return content


def format_solution_md(
    original_text: str,
    solution_body: str,
    plot_path: str | None = None,
) -> str:
    """Оборачивает решение в полный .md файл с YAML frontmatter.

    Если передан plot_path — вставляет ссылку на изображение (Obsidian-формат).
    """
    from datetime import datetime, timezone
    from pathlib import Path

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    plot_section = ""
    if plot_path:
        filename = Path(plot_path).name
        plot_section = (
            f"\n# График\n"
            f"![[{filename}]]\n"
        )

    task_text = original_text.strip()
    if task_text.startswith("# Задание"):
        task_text = task_text[len("# Задание"):].strip()

    return (
        f"---\n"
        f"status: solved\n"
        f"solved_at: {now}\n"
        f"---\n"
        f"# Задание\n"
        f"{task_text}\n\n"
        f"# Решение\n"
        f"{solution_body}\n"
        f"{plot_section}"
    )
