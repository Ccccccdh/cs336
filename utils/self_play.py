"""任务书版 Self-Play：解析新题、标准答案并做轻量质量检查。"""

from __future__ import annotations

import re

from grader.drgrpo_grader import extract_answer, grade


ANSWER_BLOCK_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

PAIR_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?(?:New\s+)?Problem"
    r"(?:\*\*)?\s*:\s*(?P<problem>.*?)"
    r"(?:^|\n|\s)(?:#+\s*)?(?:\*\*)?(?:The\s+)?Answer(?:\*\*)?"
    r"\s*(?:is\s*)?:\s*"
    r"(?P<answer>.*?)(?=(?:\\?</answer>)|$)",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

PROBLEM_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?(?:New\s+)?Problem"
    r"(?:\*\*)?\s*:\s*(?P<problem>.*?)"
    r"(?=(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?"
    r"(?:Solution|(?:The\s+)?Answer)(?:\*\*)?\s*(?:is\s*)?:"
    r"|(?:\\?</answer>)|$)",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

NEW_PROBLEM_RE = re.compile(
    r"(?:the\s+)?new\s+problem\s+(?:is\s*)?:\s*(?P<problem>.*?)"
    r"(?=(?:\\boxed)|(?:\\?</answer>)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

QUESTION_INTENT_RE = re.compile(
    r"\b(compute|find|determine|solve|evaluate|calculate|simplify|factor|"
    r"how many|how much|what|which|for what values)\b",
    flags=re.IGNORECASE,
)

CONTEXT_DEPENDENT_PHRASES = (
    "seed problem",
    "original problem",
    "given problem",
    "problem above",
    "above problem",
    "provided problem",
)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown|latex)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_problem(text: str) -> str | None:
    text = strip_code_fence(text)
    text = re.sub(
        r"^\s*(?:#+\s*)?(?:\*\*)?(?:New\s+)?Problem(?:\*\*)?\s*:\s*",
        "",
        text,
        count=1,
        flags=re.I,
    )
    text = re.split(
        r"(?:^|\n|\s)\s*(?:#+\s*)?(?:\*\*)?"
        r"(?:Solution|(?:The\s+)?Answer)(?:\*\*)?\s*(?:is\s*)?:",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    text = re.sub(
        r"\\?</?(?:think|answer)>.*$",
        "",
        text,
        flags=re.I | re.DOTALL,
    ).strip()
    text = re.sub(r"(?:\\\(|\\\[|\$)+\s*$", "", text).strip()
    text = re.sub(r"\n+\s*Inside\s*$", "", text, flags=re.I).strip()
    text = re.sub(r"\s*\[\s*answer\s*\]\s*$", "", text, flags=re.I).strip()

    if not 20 <= len(text) <= 1600:
        return None
    if "?" not in text and not QUESTION_INTENT_RE.search(text):
        return None
    return text


def _clean_answer(text: str) -> str | None:
    text = strip_code_fence(text)
    text = re.sub(
        r"^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*(?:is|:)\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\\?</?(?:think|answer)>.*$", "", text, flags=re.I | re.S)
    text = text.strip()

    if "\\boxed" in text:
        boxed = extract_answer(text)
        text = boxed.strip() if boxed else ""
    else:
        text = text.strip("$").strip()

    if not text or len(text) > 512:
        return None
    return text


def _candidate_scopes(text: str) -> list[str]:
    scopes: list[str] = []
    scopes.extend(ANSWER_BLOCK_RE.findall(text))

    if "<answer>" in text.lower():
        tail = re.split(r"<answer>", text, flags=re.I)[-1]
        tail = re.split(r"</answer>", tail, maxsplit=1, flags=re.I)[0]
        scopes.append(tail)

    if "</think>" in text.lower():
        scopes.append(re.split(r"</think>", text, flags=re.I)[-1])
    scopes.append(text)
    return scopes


def parse_problem_and_answer(text: str) -> tuple[str | None, str | None]:
    """优先解析 Problem+Answer；也允许只返回 Problem 供兼容分支补答案。"""
    if not isinstance(text, str) or not text.strip():
        return None, None

    scopes = _candidate_scopes(text)
    for scope in reversed(scopes):
        matches = list(PAIR_RE.finditer(scope))
        for match in reversed(matches):
            problem = _clean_problem(match.group("problem"))
            answer = _clean_answer(match.group("answer"))
            if problem is not None and answer is not None:
                return problem, answer

    problem_candidates: list[str] = []
    for scope in scopes:
        problem_candidates.extend(
            match.group("problem") for match in PROBLEM_RE.finditer(scope)
        )
        problem_candidates.extend(
            match.group("problem") for match in NEW_PROBLEM_RE.finditer(scope)
        )

    problem_candidates.extend(ANSWER_BLOCK_RE.findall(text))
    for candidate in reversed(problem_candidates):
        problem = _clean_problem(candidate)
        if problem is not None:
            return problem, None
    return None, None


def extract_model_answer(response: str) -> str | None:
    """从 r1-zero 解答中提取最终短答案。"""
    if not isinstance(response, str) or not response.strip():
        return None
    blocks = ANSWER_BLOCK_RE.findall(response)
    payload = blocks[-1] if blocks else response

    pair_matches = list(PAIR_RE.finditer(payload))
    if pair_matches:
        return _clean_answer(pair_matches[-1].group("answer"))

    answer = _clean_answer(payload)
    if answer is not None and len(answer) <= 256 and answer.count("\n") <= 3:
        return answer

    if "\\boxed" in response:
        boxed = extract_answer(response)
        return boxed.strip() if boxed else None
    return None


def normalize_problem(text: str) -> str:
    text = text.lower()
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".")


def validate_generated_problem(problem: str, seed_problem: str) -> str | None:
    """按任务书只做必要过滤，不拒绝合理的同题型或改数字变体。"""
    if len(problem) < 20:
        return "problem_too_short"
    if len(problem) > 1600:
        return "problem_too_long"
    lowered = problem.lower()
    if re.search(r"(?:the\s+)?answer\s*(?:is\s*)?:", problem, flags=re.I):
        return "answer_leaked_into_problem"
    if "\\boxed" in problem or re.search(r"\[\s*answer\s*\]", problem, flags=re.I):
        return "answer_leaked_into_problem"
    if any(phrase in lowered for phrase in CONTEXT_DEPENDENT_PHRASES):
        return "context_dependent"
    if normalize_problem(problem) == normalize_problem(seed_problem):
        return "copied_seed"
    return None


def format_solve_prompt(template: str, problem: str) -> str:
    return template.replace("{question}", problem)


def answers_agree(first: str, second: str) -> bool:
    try:
        return bool(grade(first, second, fast=True) or grade(second, first, fast=True))
    except Exception:
        return False
