"""Self-Play RL 主循环：出题、验证、G 组采样、GRPO 更新。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATE_SCRIPT = ROOT / "scripts" / "self_play_generate.py"
ROLLOUT_SCRIPT = ROOT / "scripts" / "self_play_rollout.py"
GRPO_TRAINER = ROOT / "trainers" / "grpo_train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Self-Play RL training")

    # 主循环和路径。
    parser.add_argument("--initial-model", default="models/grpo_fast")
    parser.add_argument("--policy-output", default="models/self_play_policy")
    parser.add_argument("--questions-path", default="data/math_sft_train.jsonl")
    parser.add_argument("--work-dir", default="data/self_play_run")
    parser.add_argument("--log-dir", default="logs/self_play_run")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--reuse-existing", action="store_true")

    # 出题和答案验证。
    parser.add_argument("--problem-prompt-file", default="prompts/self_play_problem_gen.prompt")
    parser.add_argument("--solve-prompt-file", default="prompts/r1_zero.prompt")
    parser.add_argument("--num-seeds-per-step", type=int, default=300)
    parser.add_argument("--candidates-per-seed", type=int, default=2)
    parser.add_argument("--problem-temperature", type=float, default=0.9)
    parser.add_argument("--problem-top-p", type=float, default=0.95)
    parser.add_argument("--problem-max-tokens", type=int, default=768)
    parser.add_argument("--fallback-answer-temperature", type=float, default=0.2)
    parser.add_argument("--verify-temperature", type=float, default=0.7)
    parser.add_argument("--verify-top-p", type=float, default=0.95)
    parser.add_argument("--verify-samples", type=int, default=2)
    parser.add_argument("--verify-max-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=20)

    # 每道自产题的 GRPO rollout。
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-min-tokens", type=int, default=4)
    parser.add_argument("--rollout-max-tokens", type=int, default=768)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument(
        "--keep-uniform-groups",
        action="store_true",
        help="默认仅训练 mixed groups；指定后也保存 advantage=0 的全对/全错组",
    )

    # GRPO 更新。
    parser.add_argument("--optimizer-state-path", default="models/self_play_policy_optimizer.pt")
    parser.add_argument("--initial-optimizer-state", default="models/grpo_fast_optimizer.pt")
    parser.add_argument("--epochs-per-step", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--old-logprob-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")

    # 公共推理配置。
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    print("[command]", " ".join(command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _option_exists(help_text: str, option: str) -> bool:
    return bool(re.search(rf"(?<![\w-]){re.escape(option)}(?=[\s,=]|$)", help_text))


def choose_option(help_text: str, *options: str, required: bool = True) -> str | None:
    for option in options:
        if _option_exists(help_text, option):
            return option
    if required:
        raise RuntimeError(f"grpo_train.py 不支持预期参数之一: {options}")
    return None


def add_option(
    command: list[str],
    help_text: str,
    value: Any,
    *options: str,
    required: bool = True,
) -> None:
    option = choose_option(help_text, *options, required=required)
    if option:
        command.extend([option, str(value)])


def read_grpo_help() -> str:
    result = subprocess.run(
        [sys.executable, str(GRPO_TRAINER), "-h"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"无法读取 grpo_train.py 参数:\n{result.stdout}")
    return result.stdout


def build_generate_command(
    args: argparse.Namespace,
    model: str,
    generated_path: Path,
    raw_path: Path,
    step_seed: int,
) -> list[str]:
    return [
        sys.executable, "-u", str(GENERATE_SCRIPT),
        "--model", model,
        "--questions-path", args.questions_path,
        "--problem-prompt-file", args.problem_prompt_file,
        "--solve-prompt-file", args.solve_prompt_file,
        "--output", str(generated_path),
        "--raw-output", str(raw_path),
        "--num-seeds", str(args.num_seeds_per_step),
        "--candidates-per-seed", str(args.candidates_per_seed),
        "--problem-temperature", str(args.problem_temperature),
        "--problem-top-p", str(args.problem_top_p),
        "--problem-max-tokens", str(args.problem_max_tokens),
        "--fallback-answer-temperature", str(args.fallback_answer_temperature),
        "--verify-temperature", str(args.verify_temperature),
        "--verify-top-p", str(args.verify_top_p),
        "--verify-samples", str(args.verify_samples),
        "--solve-max-tokens", str(args.verify_max_tokens),
        "--batch-size", str(args.generation_batch_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--seed", str(step_seed),
    ]


def build_rollout_command(
    args: argparse.Namespace,
    model: str,
    generated_path: Path,
    rollout_path: Path,
    step: int,
    step_seed: int,
) -> list[str]:
    command = [
        sys.executable, "-u", str(ROLLOUT_SCRIPT),
        "--model", model,
        "--input", str(generated_path),
        "--output", str(rollout_path),
        "--prompt-file", args.solve_prompt_file,
        "--group-size", str(args.group_size),
        "--temperature", str(args.rollout_temperature),
        "--top-p", str(args.rollout_top_p),
        "--min-tokens", str(args.rollout_min_tokens),
        "--max-tokens", str(args.rollout_max_tokens),
        "--batch-size", str(args.rollout_batch_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--seed", str(step_seed),
        "--grpo-step", str(step),
    ]
    if not args.keep_uniform_groups:
        command.append("--drop-uniform-groups")
    return command


def build_train_command(
    args: argparse.Namespace,
    model: str,
    rollout_path: Path,
    step_seed: int,
    help_text: str,
) -> list[str]:
    command = [sys.executable, "-u", str(GRPO_TRAINER)]
    add_option(command, help_text, model, "--model")
    add_option(command, help_text, rollout_path, "--data-path", "--rollout-path")
    add_option(command, help_text, args.policy_output, "--output-dir")
    add_option(
        command,
        help_text,
        args.optimizer_state_path,
        "--optimizer-state-path",
    )
    add_option(command, help_text, args.epochs_per_step, "--epochs")
    add_option(command, help_text, args.micro_batch_size, "--micro-batch-size")
    add_option(
        command,
        help_text,
        args.old_logprob_batch_size,
        "--old-logprob-batch-size",
        required=False,
    )
    add_option(command, help_text, args.grad_accum_steps, "--grad-accum-steps")
    add_option(command, help_text, args.learning_rate, "--learning-rate")
    add_option(command, help_text, args.max_seq_len, "--max-seq-len")
    add_option(command, help_text, args.clip_eps, "--clip-eps")
    add_option(command, help_text, args.clip_norm, "--clip-norm", required=False)
    add_option(command, help_text, args.log_every, "--log-every", required=False)
    add_option(command, help_text, step_seed, "--seed", required=False)
    if not args.no_gradient_checkpointing and _option_exists(help_text, "--gradient-checkpointing"):
        command.append("--gradient-checkpointing")
    return command


def prepare_optimizer(args: argparse.Namespace) -> None:
    destination = Path(args.optimizer_state_path)
    if destination.exists():
        print(f"[optimizer] resume: {destination}")
        return
    source = Path(args.initial_optimizer_state) if args.initial_optimizer_state else None
    if source and source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[optimizer] copied: {source} -> {destination}")
    else:
        print("[optimizer] initial state not found; start a new AdamW state")


def train(args: argparse.Namespace) -> None:
    if args.num_steps <= args.start_step:
        raise ValueError("--num-steps 必须大于 --start-step；num-steps 表示结束 step")
    for path in (GENERATE_SCRIPT, ROLLOUT_SCRIPT, GRPO_TRAINER):
        if not path.exists():
            raise FileNotFoundError(path)

    work_dir = Path(args.work_dir)
    log_dir = Path(args.log_dir)
    summary_path = work_dir / "summary.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    grpo_help = read_grpo_help()
    if not args.dry_run:
        prepare_optimizer(args)

    current_model = args.policy_output if args.start_step > 0 else args.initial_model
    print(
        f"[self-play] start={args.start_step}, end={args.num_steps}, "
        f"initial_policy={current_model}"
    )

    for step in range(args.start_step, args.num_steps):
        prefix = f"step_{step:04d}"
        generated_path = work_dir / f"{prefix}_generated.jsonl"
        raw_path = work_dir / f"{prefix}_generated_raw.jsonl"
        rollout_path = work_dir / f"{prefix}_rollouts.jsonl"
        print(f"\n{'=' * 24} SELF-PLAY STEP {step} {'=' * 24}")

        if not (args.reuse_existing and count_jsonl(generated_path) > 0):
            generate_command = build_generate_command(
                args, current_model, generated_path, raw_path, args.seed + step
            )
            run_logged(generate_command, log_dir / f"{prefix}_generate.log", args.dry_run)
        else:
            print(f"[reuse] generated data: {generated_path}")

        if args.dry_run:
            rollout_command = build_rollout_command(
                args,
                current_model,
                generated_path,
                rollout_path,
                step,
                args.seed + 100000 + step,
            )
            run_logged(rollout_command, log_dir / f"{prefix}_rollout.log", True)
            train_command = build_train_command(
                args,
                current_model,
                rollout_path,
                args.seed + 200000 + step,
                grpo_help,
            )
            run_logged(train_command, log_dir / f"{prefix}_train.log", True)
            print("[dry-run] all three stage commands validated; nothing was run")
            return

        accepted_count = count_jsonl(generated_path)
        if accepted_count == 0:
            print("[skip] no verified generated problems; policy is unchanged")
            append_jsonl(
                summary_path,
                {"step": step, "accepted": 0, "rollouts": 0, "trained": False},
            )
            continue

        if not (args.reuse_existing and rollout_path.exists()):
            rollout_command = build_rollout_command(
                args,
                current_model,
                generated_path,
                rollout_path,
                step,
                args.seed + 100000 + step,
            )
            run_logged(rollout_command, log_dir / f"{prefix}_rollout.log", False)
        else:
            print(f"[reuse] rollout data: {rollout_path}")
        rollout_count = count_jsonl(rollout_path)
        if rollout_count == 0:
            print("[skip] no mixed rollout groups; policy is unchanged")
            append_jsonl(
                summary_path,
                {
                    "step": step,
                    "accepted": accepted_count,
                    "rollouts": 0,
                    "trained": False,
                },
            )
            continue

        train_command = build_train_command(
            args,
            current_model,
            rollout_path,
            args.seed + 200000 + step,
            grpo_help,
        )
        run_logged(train_command, log_dir / f"{prefix}_train.log", args.dry_run)

        append_jsonl(
            summary_path,
            {
                "step": step,
                "input_policy": current_model,
                "output_policy": args.policy_output,
                "accepted": accepted_count,
                "rollouts": rollout_count,
                "trained": True,
            },
        )
        current_model = args.policy_output
        print(f"[step {step}] updated policy: {current_model}")

    print(f"[done] final policy: {current_model}")
    print(f"[done] summary: {summary_path}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
