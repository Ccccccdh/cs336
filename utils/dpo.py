import torch
import torch.nn.functional as F


def compute_sequence_log_probs(
    model,
    input_ids: torch.Tensor,       
    attention_mask: torch.Tensor,  
    labels: torch.Tensor,         
    response_mask: torch.Tensor,  
) -> torch.Tensor:

    # logits[t] 是模型在位置 t 对"下一个 token"的预测分布
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # 取出每个位置"目标 token"（真实下一个 token）的 log 概率
    safe_labels = labels.clamp(min=0).unsqueeze(-1)    
    token_logp = torch.gather(log_probs, -1, safe_labels)  
    token_logp = token_logp.squeeze(-1)                 

    # 只保留 response 位置，然后沿序列求和 = 整条回答的 log 概率
    valid = response_mask.float()
    seq_logp = (token_logp * valid).sum(dim=-1)   

    return seq_logp