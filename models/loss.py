import torch.nn.functional as F
import torch.nn as nn

# $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$
def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, T=100):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
    soft_loss = (alpha*T*T) * nn.KLDivLoss()(F.softmax(teacher_outputs/T, dim=1), F.log_softmax(outputs/T, dim=1))
    return hard_loss + soft_loss
