"""在固定的 PhyX 文本 held-out 集上评测物理选择题能力。

本脚本既可评测原始全量模型，也可评测合并 LoRA 后的全量模型。两次评测应
使用相同数据、采样参数和随机种子。数据来自 prepare_phyx.py 创建的自定义
分层划分，因此结果不能表述为 PhyX 官方 test 成绩。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_ANSWERS = {"A", "B", "C", "D"}
TRAINING_INSTRUCTION = (
    "Solve the physics problem. Explain the reasoning, then put only the "
    "correct option letter inside the <answer> tags."
)
ANSWER_ONLY_TEMPLATE = """A conversation between User and Assistant. The User asks a multiple-choice physics question. The Assistant must select the correct option.
User: {question}

Return exactly one <answer> tag containing exactly one of the letters A, B, C, or D. Do not provide an explanation or any other text.
Assistant:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on PhyX JSONL")
    parser.add_argument("--model", default="models/self_play_exp2_step3")
    parser.add_argument("--data-path", default="data/phyx_lora/test.jsonl")
    parser.add_argument("--output-dir", default="results/phyx_base")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--eval-mode",
        choices=["likelihood", "generation"],
        default="likelihood",
        help="主指标使用 likelihood 对 A-D 打分；generation 仅作格式诊断",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=8,
        help="likelihood 模式每批候选序列数；每道题有 4 条候选序列",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-mode",
        choices=["answer_only", "reasoning"],
        default="answer_only",
        help="主评测使用 answer_only；reasoning 仅用于检查长推理与格式",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def read_items(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            source_id = str(row.get("id") or "")
            prompt = str(row.get("prompt") or "")
            answer = str(row.get("final_answer") or "").strip().upper()
            if not source_id or not prompt or answer not in VALID_ANSWERS:
                raise ValueError(
                    f"{path}:{line_number} 缺少 id/prompt 或 final_answer 不是 A-D"
                )
            if source_id in seen_ids:
                raise ValueError(f"{path}:{line_number} 出现重复 id={source_id}")
            seen_ids.add(source_id)
            items.append(row)
            if max_examples is not None and len(items) >= max_examples:
                break
    if not items:
        raise RuntimeError("评测文件没有有效样本")
    return items


def extract_prediction(response: str) -> tuple[str | None, bool, str]:
    """返回预测字母、严格格式是否正确，以及使用的提取规则。"""

    tagged = list(
        re.finditer(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.I | re.S)
    )
    if tagged:
        content = tagged[-1].group(1).strip()
        if re.fullmatch(r"[A-Da-d]", content):
            return content.upper(), True, "strict_tag"
        match = re.search(r"(?:^|[^A-Za-z])([A-Da-d])(?:[^A-Za-z]|$)", content)
        if match:
            return match.group(1).upper(), False, "loose_tag"

    boxed = list(re.finditer(r"\\boxed\s*\{\s*([A-Da-d])\s*\}", response))
    if boxed:
        return boxed[-1].group(1).upper(), False, "boxed_letter"

    # 格式失败时仍尝试测量选择题知识，格式奖励则保持为 0。
    explicit = list(
        re.finditer(
            r"(?:answer|choice|option)(?:\s+(?:would\s+be|is))?\s*"
            r"(?::|=)?\s*(?:\\?\[|\\?\()?\s*([A-D])(?:\s*[.):\\]|\b)",
            response,
            flags=re.I,
        )
    )
    if explicit:
        return explicit[-1].group(1).upper(), False, "explicit_text"
    standalone = list(re.finditer(r"(?m)^\s*\(?([A-D])\)?[.)]?\s*$", response))
    if standalone:
        return standalone[-1].group(1).upper(), False, "standalone"
    return None, False, "missing"


def normalized_math_text(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"^\\boxed\s*\{(.*)\}$", r"\1", text, flags=re.S)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"[\s$]", "", text)
    return text.rstrip(".,;:")


def matches_correct_option_text(response: str, answer_text: Any) -> bool:
    """兼容模型在 answer/boxed 中输出选项内容而不是选项字母。"""

    expected = normalized_math_text(str(answer_text or ""))
    if not expected:
        return False
    candidates = [
        match.group(1)
        for match in re.finditer(
            r"<answer>\s*(.*?)\s*</answer>", response, flags=re.I | re.S
        )
    ]
    candidates.extend(
        match.group(1)
        for match in re.finditer(r"\\boxed\s*\{([^{}]*)\}", response, flags=re.S)
    )
    return any(normalized_math_text(candidate) == expected for candidate in candidates)


def answer_only_question(item: dict[str, Any]) -> str:
    question = str(item["question"]).strip()
    if question.endswith(TRAINING_INSTRUCTION):
        question = question[: -len(TRAINING_INSTRUCTION)].rstrip()
    return question


def prompts_for_items(
    items: list[dict[str, Any]], prompt_mode: str
) -> list[str]:
    if prompt_mode == "reasoning":
        return [str(item["prompt"]) for item in items]
    return [
        ANSWER_ONLY_TEMPLATE.replace("{question}", answer_only_question(item))
        for item in items
    ]


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for category in sorted(groups):
        category_rows = groups[category]
        count = len(category_rows)
        correct = sum(bool(row["answer_correct"]) for row in category_rows)
        result[category] = {
            "examples": count,
            "correct": correct,
            "answer_accuracy": round(correct / count, 6),
        }
        if all("format_correct" in row for row in category_rows):
            formatted = sum(bool(row["format_correct"]) for row in category_rows)
            rewarded = sum(bool(row["reward"]) for row in category_rows)
            result[category]["format_rate"] = round(formatted / count, 6)
            result[category]["strict_reward"] = round(rewarded / count, 6)
    return result


def longest_common_prefix(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    limit = min(len(sequence) for sequence in sequences)
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    return limit


def likelihood_prompt(item: dict[str, Any]) -> str:
    question = answer_only_question(item)
    return (
        "A conversation between User and Assistant. The User asks a "
        "multiple-choice physics question.\n"
        f"User: {question}\n\n"
        "Assistant: The correct option is\n<answer>"
    )


def resolve_torch_dtype(name: str, torch_module: Any) -> Any:
    normalized = name.lower()
    if normalized in {"auto", "bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    if normalized in {"float32", "fp32", "float"}:
        return torch_module.float32
    raise ValueError(f"不支持的 dtype: {name}")


def score_choices(
    items: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], float]:
    """用条件对数概率给 A/B/C/D 打分，不依赖自由生成格式。"""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("likelihood 评测需要 CUDA GPU")
    dtype = resolve_torch_dtype(args.dtype, torch)
    print(f"[init] loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print(f"[init] loading model for likelihood scoring: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()

    candidates: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        prompt = likelihood_prompt(item)
        full_ids = [
            tokenizer.encode(prompt + letter, add_special_tokens=False)
            for letter in sorted(VALID_ANSWERS)
        ]
        prefix_length = longest_common_prefix(full_ids)
        if prefix_length <= 0:
            raise RuntimeError(f"样本 {item.get('id')} 无法确定公共 prompt token")
        for letter, token_ids in zip(sorted(VALID_ANSWERS), full_ids):
            if prefix_length >= len(token_ids):
                raise RuntimeError(f"样本 {item.get('id')} 的候选 {letter} 没有后缀 token")
            if len(token_ids) > args.max_seq_len:
                removed = len(token_ids) - args.max_seq_len
                token_ids = token_ids[removed:]
                candidate_start = prefix_length - removed
            else:
                candidate_start = prefix_length
            if candidate_start <= 0:
                raise RuntimeError(
                    f"样本 {item.get('id')} 超过 max-seq-len，无法保留评分上下文"
                )
            candidates.append(
                {
                    "item_index": item_index,
                    "letter": letter,
                    "token_ids": token_ids,
                    "candidate_start": candidate_start,
                }
            )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer 没有 pad_token_id 或 eos_token_id")

    scores: list[dict[str, float]] = [dict() for _ in items]
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(candidates), args.score_batch_size):
            batch = candidates[start : start + args.score_batch_size]
            max_length = max(len(candidate["token_ids"]) for candidate in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                fill_value=pad_id,
                dtype=torch.long,
                device="cuda",
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, candidate in enumerate(batch):
                ids = torch.tensor(
                    candidate["token_ids"], dtype=torch.long, device="cuda"
                )
                input_ids[row_index, : len(ids)] = ids
                attention_mask[row_index, : len(ids)] = 1

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row_index, candidate in enumerate(batch):
                token_ids = candidate["token_ids"]
                candidate_start = candidate["candidate_start"]
                score = 0.0
                for token_index in range(candidate_start, len(token_ids)):
                    score += float(
                        log_probs[
                            row_index,
                            token_index - 1,
                            token_ids[token_index],
                        ].item()
                    )
                scores[candidate["item_index"]][candidate["letter"]] = score

            completed = min(start + len(batch), len(candidates))
            if completed == len(candidates) or completed % max(4, args.score_batch_size * 10) == 0:
                print(f"[score] {completed}/{len(candidates)} candidate sequences")

    elapsed = time.time() - started
    rows: list[dict[str, Any]] = []
    for index, (item, option_scores) in enumerate(zip(items, scores)):
        if set(option_scores) != VALID_ANSWERS:
            raise RuntimeError(f"样本 {item.get('id')} 的 A-D 分数不完整: {option_scores}")
        prediction = max(option_scores, key=option_scores.get)
        ground_truth = str(item["final_answer"]).upper()
        rows.append(
            {
                "index": index,
                "id": item["id"],
                "category": item.get("category", "Unknown"),
                "subfield": item.get("subfield", ""),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "answer_correct": prediction == ground_truth,
                "choice_logprobs": {
                    letter: round(option_scores[letter], 6)
                    for letter in sorted(VALID_ANSWERS)
                },
                "prompt": likelihood_prompt(item),
            }
        )

    del model
    torch.cuda.empty_cache()
    return rows, elapsed


def generate(
    items: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], float]:
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    print(f"[init] loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples.jsonl"
    prompts = prompts_for_items(items, args.prompt_mode)
    records: list[dict[str, Any]] = []
    total_seconds = 0.0

    with sample_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(items), args.batch_size):
            end = min(start + args.batch_size, len(items))
            print(f"[generate] {start + 1}-{end} / {len(items)}")
            begin = time.time()
            outputs = llm.generate(prompts[start:end], sampling_params)
            total_seconds += time.time() - begin

            for offset, output in enumerate(outputs):
                index = start + offset
                item = items[index]
                completion = output.outputs[0]
                response = completion.text
                prediction, format_correct, extraction = extract_prediction(response)
                ground_truth = str(item["final_answer"]).upper()
                if prediction is None and matches_correct_option_text(
                    response, item.get("answer_text")
                ):
                    prediction = ground_truth
                    extraction = "correct_option_text"
                answer_correct = prediction == ground_truth
                record = {
                    "index": index,
                    "id": item["id"],
                    "category": item.get("category", "Unknown"),
                    "subfield": item.get("subfield", ""),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "answer_correct": answer_correct,
                    "format_correct": format_correct,
                    "reward": bool(answer_correct and format_correct),
                    "extraction": extraction,
                    "prompt": prompts[index],
                    "response": response,
                    "response_tokens": len(completion.token_ids),
                    "finish_reason": getattr(completion, "finish_reason", None),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return records, total_seconds


def write_metrics(
    rows: list[dict[str, Any]], total_seconds: float, args: argparse.Namespace
) -> dict[str, Any]:
    count = len(rows)
    answer_correct = sum(bool(row["answer_correct"]) for row in rows)
    per_category = category_metrics(rows)
    macro_accuracy = sum(
        item["answer_accuracy"] for item in per_category.values()
    ) / len(per_category)
    metrics = {
        "dataset": "Cloudriver/PhyX custom category-stratified held-out split",
        "data_path": args.data_path,
        "model": args.model,
        "eval_mode": args.eval_mode,
        "prompt_mode": args.prompt_mode,
        "num_examples": count,
        "answer_correct": answer_correct,
        "answer_accuracy": round(answer_correct / count, 6),
        "macro_category_accuracy": round(macro_accuracy, 6),
        "prediction_distribution": dict(
            sorted(Counter(str(row["prediction"]) for row in rows).items())
        ),
        "per_category": per_category,
        "total_evaluation_seconds": round(total_seconds, 2),
    }
    if args.eval_mode == "generation":
        format_correct = sum(bool(row["format_correct"]) for row in rows)
        rewarded = sum(bool(row["reward"]) for row in rows)
        metrics.update(
            {
                "format_correct": format_correct,
                "format_rate": round(format_correct / count, 6),
                "strict_reward_mean": round(rewarded / count, 6),
                "missing_prediction": sum(row["prediction"] is None for row in rows),
                "extraction_distribution": dict(
                    sorted(Counter(row["extraction"] for row in rows).items())
                ),
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                },
            }
        )
    else:
        metrics["scoring"] = {
            "method": "conditional_log_likelihood_over_A_B_C_D",
            "score_batch_size": args.score_batch_size,
            "max_seq_len": args.max_seq_len,
        }
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] results: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_tokens <= 0 or args.score_batch_size <= 0:
        raise ValueError("batch-size、score-batch-size 和 max-tokens 必须为正数")
    items = read_items(Path(args.data_path), args.max_examples)
    print(f"[data] loaded {len(items)} examples from {args.data_path}")
    if args.eval_mode == "likelihood":
        rows, total_seconds = score_choices(items, args)
    else:
        rows, total_seconds = generate(items, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_metrics(rows, total_seconds, args)


if __name__ == "__main__":
    main()
