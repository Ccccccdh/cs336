from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run full GRPO training"
    )

    p.add_argument(
        "--initial-model",
        type=str,
        default=str(ROOT / "models" / "dpo_beta01"),
    )
    p.add_argument(
        "--questions-path",
        type=str,
        default=str(
            ROOT / "data" / "math_sft_train.jsonl"
        ),
    )
    p.add_argument(
        "--policy-output",
        type=str,
        default=str(ROOT / "models" / "grpo_full"),
    )
    p.add_argument(
        "--optimizer-state-path",
        type=str,
        default=str(
            ROOT / "models" / "grpo_full_optimizer.pt"
        ),
    )
    p.add_argument(
        "--rollout-dir",
        type=str,
        default=str(
            ROOT / "data" / "grpo_full_rollouts"
        ),
    )

    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--start-step", type=int, default=0)

    p.add_argument(
        "--rollout-batch-size",
        type=int,
        default=32,
    )
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--rollout-engine-batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--epochs-per-rollout", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--old-logprob-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--clip-norm", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--normalize-advantages",
        action="store_true",
    )

    return p.parse_args()


def run_command(command):
    print()
    print("[command]", shlex.join(command))
    print()

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    args = parse_args()

    if not 0 <= args.start_step < args.num_steps:
        raise ValueError(
            "start-step 必须满足 "
            "0 <= start-step < num-steps"
        )

    policy_output = Path(args.policy_output)
    optimizer_path = Path(args.optimizer_state_path)
    rollout_dir = Path(args.rollout_dir)

    rollout_dir.mkdir(parents=True, exist_ok=True)
    policy_output.parent.mkdir(parents=True, exist_ok=True)
    optimizer_path.parent.mkdir(parents=True, exist_ok=True)

    if args.start_step > 0:
        if not policy_output.exists():
            raise FileNotFoundError(
                f"续训模型不存在: {policy_output}"
            )
        if not optimizer_path.exists():
            raise FileNotFoundError(
                f"续训 optimizer 不存在: {optimizer_path}"
            )

    for step in range(args.start_step, args.num_steps):
        if step == 0:
            current_model = args.initial_model
        else:
            current_model = args.policy_output

        rollout_path = (
            rollout_dir
            / f"step_{step:04d}.jsonl"
        )

        print()
        print("=" * 70)
        print(
            f"[GRPO] outer step "
            f"{step + 1}/{args.num_steps}"
        )
        print(f"[GRPO] current policy: {current_model}")
        print("=" * 70)

        rollout_command = [
            sys.executable,
            "scripts/grpo_rollout.py",
            "--model",
            current_model,
            "--questions-path",
            args.questions_path,
            "--num-questions",
            str(args.rollout_batch_size),
            "--group-size",
            str(args.group_size),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--grpo-step",
            str(step),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.rollout_engine_batch_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output",
            str(rollout_path),
        ]

        if args.normalize_advantages:
            rollout_command.append(
                "--normalize-advantages"
            )

        run_command(rollout_command)

        train_command = [
            sys.executable,
            "trainers/grpo_train.py",
            "--model",
            current_model,
            "--rollout-path",
            str(rollout_path),
            "--output-dir",
            args.policy_output,
            "--optimizer-state-path",
            args.optimizer_state_path,
            "--epochs",
            str(args.epochs_per_rollout),
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--old-logprob-batch-size",
            str(args.old_logprob_batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--max-seq-len",
            str(args.max_seq_len),
            "--clip-eps",
            str(args.clip_eps),
            "--clip-norm",
            str(args.clip_norm),
            "--gradient-checkpointing",
            "--seed",
            str(args.seed + step),
            "--log-every",
            "1",
        ]

        run_command(train_command)

        print(
            f"[GRPO] outer step {step} 完成，"
            f"policy={policy_output}"
        )

    print()
    print("[done] GRPO 全量训练完成")
    print(f"[done] 最终模型: {policy_output}")


if __name__ == "__main__":
    main()