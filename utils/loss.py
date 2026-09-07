import torch
import torch.nn.functional as F


def compute_sft_loss_and_entropy(
    model,
    input_ids: torch.Tensor,       
    attention_mask: torch.Tensor, 
    labels: torch.Tensor,         
    response_mask: torch.Tensor, 
):

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp(min=0).unsqueeze(-1) 
    token_logp = torch.gather(log_probs, -1, safe_labels)   
    token_logp = token_logp.squeeze(-1)                     

    valid = response_mask.float()
    loss = -(token_logp * valid).sum() / valid.sum().clamp(min=1)

    probs = log_probs.exp()
    per_token_entropy = -(probs * log_probs).sum(dim=-1) 
    entropy = (per_token_entropy * valid).sum() / valid.sum().clamp(min=1)

    return loss, entropy