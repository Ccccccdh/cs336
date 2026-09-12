from __future__ import annotations

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:

    # 按问题组计算 GRPO advantage。
    if rewards.ndim != 2:
        raise ValueError(
            "rewards 必须是 [num_questions, group_size]，"
            f"实际 shape={tuple(rewards.shape)}"
        )

    if rewards.shape[1] == 0:
        raise ValueError("group_size 不能为 0")

    rewards = rewards.to(dtype=torch.float32)

    group_mean = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - group_mean

    if normalize:
        # 使用总体标准差，避免 group_size=1 时出现 NaN。
        group_std = rewards.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        )
        advantages = advantages / (group_std + eps)

    return advantages


def compute_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    
    # 计算每个目标 token 的 log probability。
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids 必须是二维张量，"
            f"实际 shape={tuple(input_ids.shape)}"
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask 与 input_ids 的 shape 必须相同"
        )

    if labels.shape != input_ids.shape:
        raise ValueError(
            "labels 与 input_ids 的 shape 必须相同"
        )

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    valid_labels = labels >= 0

    # gather 不接受 -100，因此无效位置暂时替换为 token 0。
    safe_labels = labels.masked_fill(~valid_labels, 0)

    target_logits = torch.gather(
        logits,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    # log p(token) = target_logit - logsumexp(all_logits)
    #
    # 不构造完整 log_softmax 张量，可少保留一个
    # [B, T, vocab_size] 的大张量。
    log_normalizer = torch.logsumexp(
        logits,
        dim=-1,
    )
    token_log_probs = target_logits - log_normalizer

    # prompt 和 padding 位置明确置零。
    token_log_probs = token_log_probs.masked_fill(
        ~valid_labels,
        0.0,
    )

    return token_log_probs


def compute_grpo_clip_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
):

    # 计算 token-level GRPO-Clip loss。
    if current_log_probs.ndim != 2:
        raise ValueError(
            "current_log_probs 必须是 "
            "[batch_size, sequence_length]"
        )

    if old_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            "old_log_probs 与 current_log_probs "
            "的 shape 必须相同"
        )

    if response_mask.shape != current_log_probs.shape:
        raise ValueError(
            "response_mask 与 current_log_probs "
            "的 shape 必须相同"
        )

    batch_size = current_log_probs.shape[0]

    if advantages.ndim == 1:
        if advantages.shape[0] != batch_size:
            raise ValueError(
                "advantages 的 batch size 不正确"
            )
        advantages = advantages.unsqueeze(1)
    elif advantages.shape != (batch_size, 1):
        raise ValueError(
            "advantages 必须是 [batch_size] "
            "或 [batch_size, 1]"
        )

    if clip_eps <= 0:
        raise ValueError("clip_eps 必须大于 0")

    old_log_probs = old_log_probs.detach().to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    advantages = advantages.detach().to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    response_mask = response_mask.to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )

    valid_token_count = response_mask.sum()
    if valid_token_count.item() == 0:
        raise ValueError(
            "当前 micro-batch 没有有效 response token"
        )

    # ratio = π_theta / π_old
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped_objective = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_eps,
        max=1.0 + clip_eps,
    )
    clipped_objective = clipped_ratio * advantages

    # 对正负 advantage 都采用更保守的目标。
    token_objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    token_loss = -token_objective

    loss = (
        token_loss * response_mask
    ).sum() / valid_token_count

    # 训练日志指标，不参与反向传播。
    with torch.no_grad():
        ratio_mean = (
            ratio * response_mask
        ).sum() / valid_token_count

        outside_clip = (
            (ratio < 1.0 - clip_eps)
            | (ratio > 1.0 + clip_eps)
        ).to(response_mask.dtype)

        clip_fraction = (
            outside_clip * response_mask
        ).sum() / valid_token_count

        # 非负的近似 KL：
        # exp(x) - 1 - x >= 0
        approx_kl_per_token = (
            ratio - 1.0 - log_ratio
        )
        approx_kl = (
            approx_kl_per_token * response_mask
        ).sum() / valid_token_count

        metrics = {
            "ratio_mean": ratio_mean.item(),
            "clip_fraction": clip_fraction.item(),
            "approx_kl": approx_kl.item(),
            "valid_tokens": int(valid_token_count.item()),
        }

    return loss, metrics