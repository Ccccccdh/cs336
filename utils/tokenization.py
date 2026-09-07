from __future__ import annotations

import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
    max_length: int | None = None,
):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_input_ids = []
    all_labels = []
    all_masks = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        if max_length is not None:
            if len(prompt_ids) >= max_length:
                prompt_ids = prompt_ids[: max_length - 1]
            response_ids = response_ids[: max_length - len(prompt_ids)]

        seq_ids = prompt_ids + response_ids
        seq_len = len(seq_ids)

        labels = [-100] * seq_len
        response_mask = [False] * seq_len

        start = len(prompt_ids) - 1
        end = seq_len - 1  
        for t in range(start, end):
            labels[t] = seq_ids[t + 1]
            response_mask[t] = True

        all_input_ids.append(seq_ids)
        all_labels.append(labels)
        all_masks.append(response_mask)

    batch_len = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id

    def pad_rows(rows, pad_value):
        out = []
        for row in rows:
            out.append(row + [pad_value] * (batch_len - len(row)))
        return out

    return (
        torch.tensor(pad_rows(all_input_ids, pad_id), dtype=torch.long),
        torch.tensor(pad_rows(all_labels, -100), dtype=torch.long),
        torch.tensor(pad_rows(all_masks, False), dtype=torch.bool),
    )