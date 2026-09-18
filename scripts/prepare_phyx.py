"""把 Cloudriver/PhyX 整理成文本 LoRA 训练、验证和测试 JSONL。

设计原则：
1. 使用 with_steps/data_with_steps，而不是包含 Base64 图片的评测配置；
2. 文本模型看不到图片，因此把 image_caption 作为题面的一部分；
3. 按 category 分层并固定随机种子，创建互不重叠的课程实验划分；
4. steps 仅作为数据集提供的解题步骤使用，不声称它们是人工完整证明；
5. 监督答案固定为 <answer>A/B/C/D</answer>，便于训练和评测。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "prompts" / "r1_zero.prompt"
VALID_ANSWERS = {"A", "B", "C", "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 PhyX 文本 LoRA 数据")
    parser.add_argument("--dataset", default="Cloudriver/PhyX")
    parser.add_argument("--dataset-config", default="with_steps")
    parser.add_argument("--source-split", default="data_with_steps")
    parser.add_argument("--output-dir", default="data/phyx_lora")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--caption-max-chars",
        type=int,
        default=2400,
        help="限制图像描述长度；0 表示不截断",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="仅用于冒烟测试；正式准备数据时不要设置",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return shortened + " ..."


def normalize_answer(value: Any) -> str | None:
    text = clean_text(value).upper()
    match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", text)
    if match:
        return match.group(1)
    return text if text in VALID_ANSWERS else None


def option_letter(option: str) -> str | None:
    match = re.match(r"^\s*(?:\(([A-Da-d])\)|([A-Da-d]))\s*[:.)-]", option)
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def option_body(option: str) -> str:
    return re.sub(
        r"^\s*(?:\([A-Da-d]\)|[A-Da-d])\s*[:.)-]\s*",
        "",
        option,
        count=1,
    ).strip()


def resolve_answer_text(options: list[str], answer: str) -> str | None:
    for index, raw_option in enumerate(options):
        option = clean_text(raw_option)
        letter = option_letter(option)
        if letter == answer:
            return option_body(option)
        if letter is None and index == ord(answer) - ord("A"):
            return option
    return None


def format_options(options: list[str]) -> str:
    formatted: list[str] = []
    for index, raw_option in enumerate(options):
        option = clean_text(raw_option)
        if not option:
            continue
        letter = option_letter(option)
        if letter:
            formatted.append(f"{letter}. {option_body(option)}")
        else:
            formatted.append(f"{chr(ord('A') + index)}. {option}")
    return "\n".join(formatted)


def build_question(row: dict[str, Any], caption_max_chars: int) -> str:
    description = clean_text(
        row.get("question_description_simplified")
        or row.get("question_description")
    )
    question = clean_text(row.get("question"))
    caption = truncate_text(clean_text(row.get("image_caption")), caption_max_chars)
    options = format_options(list(row.get("options") or []))

    sections: list[str] = []
    if description:
        sections.append(f"Problem description:\n{description}")
    if caption:
        sections.append(f"Diagram description:\n{caption}")
    if question and question.lower() not in description.lower():
        sections.append(f"Question:\n{question}")
    if options:
        sections.append(f"Options:\n{options}")
    sections.append(
        "Solve the physics problem. Explain the reasoning, then put only the "
        "correct option letter inside the <answer> tags."
    )
    return "\n\n".join(sections)


def build_response(steps: Iterable[Any], answer: str, answer_text: str) -> str:
    cleaned_steps = [clean_text(step) for step in steps]
    cleaned_steps = [step for step in cleaned_steps if step]
    if cleaned_steps:
        reasoning = "\n".join(cleaned_steps)
    else:
        reasoning = "Use the stated physical quantities and relations to compare the options."
    conclusion = f"Therefore the matching choice is {answer}: {answer_text}."
    return f"{reasoning}\n{conclusion}\n</think> <answer>{answer}</answer>"


def convert_row(
    row: dict[str, Any],
    prompt_template: str,
    caption_max_chars: int,
) -> tuple[dict[str, Any] | None, str | None]:
    source_id = clean_text(row.get("id"))
    answer = normalize_answer(row.get("answer"))
    options = [clean_text(option) for option in (row.get("options") or [])]
    options = [option for option in options if option]
    if not source_id:
        return None, "missing_id"
    if answer not in VALID_ANSWERS:
        return None, "invalid_answer"
    if len(options) < 4:
        return None, "missing_options"
    answer_text = resolve_answer_text(options, answer)
    if not answer_text:
        return None, "answer_option_not_found"

    question = build_question(row, caption_max_chars)
    if not question:
        return None, "empty_question"
    prompt = prompt_template.replace("{question}", question)
    response = build_response(row.get("steps") or [], answer, answer_text)
    return {
        "id": source_id,
        "question": question,
        "final_answer": answer,
        "answer_text": answer_text,
        "prompt": prompt,
        "response": response,
        "category": clean_text(row.get("category")) or "Unknown",
        "subfield": clean_text(row.get("subfield")),
        "reasoning_type": list(row.get("reasoning_type") or []),
        "formulations": list(row.get("formulations") or []),
        "steps_count": len([x for x in (row.get("steps") or []) if clean_text(x)]),
        "data_source": "Cloudriver/PhyX:with_steps/data_with_steps",
    }, None


def stratified_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for category in sorted(groups):
        category_rows = list(groups[category])
        # 每个类别使用独立且稳定的 RNG，避免类别遍历顺序影响划分。
        category_seed = seed + sum((index + 1) * ord(ch) for index, ch in enumerate(category))
        random.Random(category_seed).shuffle(category_rows)
        size = len(category_rows)
        validation_size = round(size * validation_ratio)
        test_size = round(size * (1.0 - train_ratio - validation_ratio))
        if size >= 3:
            validation_size = max(1, validation_size)
            test_size = max(1, test_size)
        if validation_size + test_size >= size:
            raise ValueError(f"类别 {category!r} 的划分比例没有留下训练样本")

        splits["validation"].extend(category_rows[:validation_size])
        splits["test"].extend(
            category_rows[validation_size : validation_size + test_size]
        )
        splits["train"].extend(category_rows[validation_size + test_size :])

    for offset, split_name in enumerate(("train", "validation", "test")):
        random.Random(seed + 10_000 + offset).shuffle(splits[split_name])
    return splits


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_categories(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def verify_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    id_sets = {
        name: {str(row["id"]) for row in rows}
        for name, rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise RuntimeError(f"{left}/{right} 出现 {len(overlap)} 个重复 ID")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio 必须在 (0, 1) 内")
    if not (0.0 < args.validation_ratio < 1.0):
        raise ValueError("--validation-ratio 必须在 (0, 1) 内")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("训练比例与验证比例之和必须小于 1")

    from datasets import Image, load_dataset

    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    if "{question}" not in prompt_template:
        raise ValueError("prompt 模板必须包含 {question}")

    print(
        f"[data] loading {args.dataset}/{args.dataset_config}:"
        f"{args.source_split}"
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.source_split,
    )
    if "image" in dataset.features and isinstance(dataset.features["image"], Image):
        dataset = dataset.cast_column("image", Image(decode=False))

    accepted: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for index, row in enumerate(dataset):
        if args.max_examples is not None and index >= args.max_examples:
            break
        converted, rejection = convert_row(
            dict(row), prompt_template, args.caption_max_chars
        )
        if converted is None:
            rejections[rejection or "unknown"] += 1
            continue
        if converted["id"] in seen_ids:
            rejections["duplicate_id"] += 1
            continue
        seen_ids.add(converted["id"])
        accepted.append(converted)

    if len(accepted) < 3:
        raise RuntimeError("有效 PhyX 样本少于 3 条，无法划分数据")

    splits = stratified_split(
        accepted,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    verify_disjoint(splits)

    output_dir = Path(args.output_dir)
    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    manifest = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": 1.0 - args.train_ratio - args.validation_ratio,
        },
        "source_rows_seen": len(accepted) + sum(rejections.values()),
        "accepted": len(accepted),
        "rejections": dict(sorted(rejections.items())),
        "splits": {
            split_name: {
                "examples": len(rows),
                "categories": count_categories(rows),
            }
            for split_name, rows in splits.items()
        },
        "id_overlap_verified_zero": True,
        "evaluation_note": (
            "These are deterministic category-stratified custom held-out splits "
            "created from PhyX data_with_steps, not the official untouched PhyX test."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[stats]")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[save] {output_dir / 'train.jsonl'}")
    print(f"[save] {output_dir / 'validation.jsonl'}")
    print(f"[save] {output_dir / 'test.jsonl'}")
    print(f"[save] {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
