import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, class_logits, domain_temperature=1.0, class_temperature=10.0):
    if domain_out is None or class_logits is None or domain_out.size(0) == 0 or class_logits.size(0) == 0:
        print("❌ Empty input to get_target_share_weight. Returning default weights.")
        return torch.ones(64, device='cuda' if torch.cuda.is_available() else 'cpu') * 1e-2

    class_prob = F.softmax(class_logits / class_temperature, dim=1)
    class_entropy = -torch.sum(class_prob * torch.log(class_prob + 1e-6), dim=1)

    # Normalize entropy
    if class_entropy.max() == class_entropy.min():
        class_entropy_norm = torch.zeros_like(class_entropy)
    else:
        class_entropy_norm = (class_entropy - class_entropy.min()) / (class_entropy.max() - class_entropy.min() + 1e-6)

    domain_prob = F.softmax(domain_out / domain_temperature, dim=1)
    domain_entropy = -torch.sum(domain_prob * torch.log(domain_prob + 1e-6), dim=1)

    weight = (1 - class_entropy_norm) * domain_entropy
    weight = torch.clamp(weight, min=1e-2)  # prevent zero weights
    return weight



def normalize_weight(x):
    if x.numel() == 0:
        print("⚠️ normalize_weight received empty tensor. Returning uniform fallback.")
        return torch.ones(1, device=x.device)  # Safe fallback
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-6)


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)